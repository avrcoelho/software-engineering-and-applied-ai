import tf, { log } from "@tensorflow/tfjs-node";

async function trainModule(inputXs, outputYs) {
  const model = tf.sequential();

  // first network layuer
  // input of 7 positions (normalized age + 3 colors + 3 locations)

  // 80 neuronios = aqui coloquei tudo isso porque tem pouca base de treino
  // quanto mais neuronios, mais complexidade a rede pode aprender
  // e consequentimente, maius processamento ela vai usar

  // A ReLU age como um filtro:
  // é como se ela deixasse somente os dados interessantes seguirem viagen na rede
  // Se a informaão chegou nesse neuronio é positiva, passa para frente!
  // se for zero ou negativa, pode jogar fora, não vai servir para nada
  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
  );

  // Saída: 3 neuronios
  // um para cada categoria (premium, medium, basic)

  // activation: softmax normaliza a saída em probabilidades
  model.add(
    tf.layers.dense({
      units: 3,
      activation: "softmax",
    }),
  );

  // compilando o modelo
  // optimizer Adam (Adaptive Moment Estimation)
  // É um treinador pessoal moderno para redes neurais
  // ajusta os pesos de forma eficiente e inteligente
  //  aprender com historico de erros e acertos

  // loss: categoricalCrossentropy
  // ele compara o que o modelo "acha" (os scores de cada categoria)
  // com a resposta certa
  // a categoria premium sera sempre  [1, 0, 0]

  // Quanto mais distante da previsão do modelo da resposta correta
  // maior o erro (loss)
  // Exemplo classico: classificacão de imagens, recomendacão, categorizacão de usuáriuo
  // Qualquer cosa em que a resposta certa é "apenas uma entre várias possiveis"
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // treinamento do modelo
  // verbose: desabilta i log interno (e usa só o callback)
  // epochs: quantidade de vezes que vai rodar no dataset
  // shuffle: embaralha os dados, para evitar viés
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(5)}`);
      },
    },
  });

  return model;
}

async function predict(model, pessoa) {
  // transformar o array js para o tensor (tfjs)
  const tfinput = tf.tensor2d(pessoa);

  // faz a predicao (output será um vetor de 3 probabilidades)
  const pred = model.predict(tfinput);
  const predArray = await pred.array();
  return predArray[0].map((prob, index) => ({
    prob,
    index,
  }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

//  quanto mais dados melhor
// assim o algoritmo consegue entender melhor os padrões complexos dos dados
const model = await trainModule(inputXs, outputYs);

const pessoa = {
  nome: "josé",
  idade: 1,
  cor: "azul",
  localizacao: "Curitiba",
};
// normaliozando a idade da nova pessoa usando o mesmo padrão de treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2

const pessoaNormalizada = [
  [
    0.2, // idade normalizada
    1, // azul
    0, // vermelho
    0, // verde
    1, // São Paulo
    0, // Rio
    0, // Jundiaí
  ],
];

const predictions = await predict(model, pessoaNormalizada);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}`)
  .join(`\n`);

console.log(results);
