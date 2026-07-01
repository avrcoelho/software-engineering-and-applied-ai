import { ChatGoogle } from "@langchain/google";
import { config, prompts, type ModelConfig } from "../config.ts";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { createAgent } from "langchain";
import { getMCPTools } from "./mcpService.ts";
import { PromptTemplate } from "@langchain/core/prompts";

export type GuardrailResult = {
  safe: boolean;
  reason?: string;
  score?: number;
  analysis?: string;
};

export class GeminiService {
  private config: ModelConfig;
  private llmClient: ChatGoogle;
  private safeGuardModel: ChatGoogle;
  private fsAgent: ReturnType<typeof createAgent> | null = null;

  constructor(configOverride?: ModelConfig) {
    this.config = configOverride ?? config;
    this.llmClient = this.#createChatModel(this.config.models[0]);
    this.safeGuardModel = this.#createChatModel(this.config.guardrailsModel);
  }

  #createChatModel(modelName: string): ChatGoogle {
    return new ChatGoogle({
      apiKey: this.config.apiKey,
      model: modelName,
      temperature: this.config.temperature,
    });
  }

  async generate(systemPrompt: string, userPrompt: string): Promise<string> {
    if (!this.fsAgent) {
      const tools = await getMCPTools();
      this.fsAgent = createAgent({
        model: this.llmClient,
        tools,
      });
    }

    const messages = [
      new SystemMessage(systemPrompt),
      new HumanMessage(userPrompt),
    ];

    const response = await this.fsAgent.invoke({ messages });
    const content = String(response.messages.at(-1)?.text ?? "");

    return content;
  }

  async checkGuardRails(userInput: string, enabled: boolean = true) {
    if (!enabled) {
      return { safe: true, reason: "Guardrails disabled" };
    }

    const template = PromptTemplate.fromTemplate(prompts.guardrails);
    const input = await template.format({
      USER_INPUT: userInput,
    });
    const response = await this.safeGuardModel.invoke([
      {
        role: "user",
        content: input,
      },
    ]);
    const result = response.text.trim();
    const isUnsafe = result.toUpperCase().startsWith("UNSAFE");
    if (isUnsafe) {
      return {
        safe: false,
        reason: "Prompt Injection detected by safeguard model",
        analysis: result,
      };
    }

    return {
      safe: true,
      analysis: result,
    };
  }
}
