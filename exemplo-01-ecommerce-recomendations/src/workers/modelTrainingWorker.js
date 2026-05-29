import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};
let _model = null;
const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

// Normalize continuous values (price, age) to 0-1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56
const normalize = (val, min, max) => (val - min) / (max - min || 1);

function makeContext(products, users) {
  const ages = users.map((user) => user.age);
  const prices = products.map((item) => item.price);

  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);

  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  const colors = [...new Set(products.map((item) => item.color))];
  const categories = [...new Set(products.map((item) => item.category))];

  const colorsIndex = Object.fromEntries(
    colors.map((color, index) => [color, index]),
  );
  const categoriesIndex = Object.fromEntries(
    categories.map((category, index) => [category, index]),
  );

  // Computar a media de idade dos compradores por produto
  // Ajuda a personalizar
  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};

  users.forEach((user) => {
    user.purchases.forEach((purchase) => {
      ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age;
      ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1;
    });
  });

  const productAvgAgeNorm = Object.fromEntries(
    products.map((product) => {
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge;
      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );

  return {
    products,
    users,
    colorsIndex,
    categoriesIndex,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    nunCategories: categories.length,
    numColors: colors.length,
    // price + age + colors + categories
    dimensions: 2 + categories.length + colors.length,
    productAvgAgeNorm,
  };
}

const oneHotWheighted = (index, length, weight) =>
  tf.oneHot(index, length).cast("float32").mul(weight);

function encodeProduct(product, context) {
  // normalizando dados para ficar se 0 a 1
  // aplicar o peso na recomendação
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) *
      WEIGHTS.price,
  ]);

  const age = tf.tensor1d([
    (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age,
  ]);

  const category = oneHotWheighted(
    context.categoriesIndex[product.category],
    context.nunCategories,
    WEIGHTS.category,
  );

  const color = oneHotWheighted(
    context.colorsIndex[product.color],
    context.numColors,
    WEIGHTS.color,
  );

  return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, context) {
  if (user.purchases.length) {
    return tf
      .stack(user.purchases.map((purchase) => encodeProduct(purchase, context)))
      .mean(0)
      .reshape([1, context.dimensions]);
  }

  return tf
    .concat1d([
      tf.zeros([1]), // preco é ignorado
      tf.tensor1d([
        normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age,
      ]),
      tf.zeros([context.nunCategories]), // categoria ignorada
      tf.zeros([context.numColors]), // cores ignoradas
    ])
    .reshape([1, context.dimensions]);
}

function createTraningData(context) {
  const inputs = [];
  const labels = [];
  context.users
    .filter((user) => user.purchases.length)
    .forEach((user) => {
      const userVector = encodeUser(user, context).dataSync();
      context.products.forEach((product) => {
        const productVector = encodeProduct(product, context).dataSync();
        const label = user.purchases.some(
          (purchase) => purchase.name === product.name,
        );

        // combinar user + product
        inputs.push([...userVector, ...productVector]);
        labels.push(label);
      });
    });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    inputsDimention: context.dimensions * 2,
    // tamanho = userVector + productVector
  };
}

// Configuracao da rede neural
async function configureNeuralNetAndTrain(trainingData) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [trainingData.inputsDimention],
      units: 128,
      activation: "relu",
    }),
  );
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    }),
  );
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    }),
  );
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
    }),
  );
  model.compile({
    optimizer: tf.train.adam(),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainingData.xs, trainingData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });
  return model;
}

async function trainModel({ users }) {
  console.log("Training model with users:", users);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 50 },
  });
  const products = await (await fetch("/data/products.json")).json();
  const context = makeContext(products, users);
  context.productVectors = products.map((product) => {
    return {
      name: product.name,
      meta: {
        ...product,
      },
      vector: encodeProduct(product, context).dataSync(),
    };
  });
  _globalCtx = context;

  //   treinar os dados
  const trainingData = createTraningData(context);
  _model = await configureNeuralNetAndTrain(trainingData);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 },
  });
  postMessage({ type: workerEvents.trainingComplete });
}
function recommend(user, ctx) {
  if (!_model) {
    return;
  }
  const context = _globalCtx;
  const userVector = encodeUser(user, context).dataSync();
  const inputs = context.productVectors.map(({ vector }) => {
    return [...userVector, ...vector];
  });
  const inputTensor = tf.tensor2d(inputs);
  const predictions = _model.predict(inputTensor);
  const scores = predictions.dataSync();
  const recommendations = context.productVectors.map((product, index) => ({
    ...product.meta,
    name: product.name,
    score: scores[index],
  }));
  const sortedItems = recommendations.sort((a, b) => b.score - a.score);
  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations: sortedItems,
  });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
