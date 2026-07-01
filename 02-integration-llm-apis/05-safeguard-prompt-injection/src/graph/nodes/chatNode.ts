import type { GraphState } from "../state.ts";
import { AIMessage } from "@langchain/core/messages";
import { GeminiService } from "../../services/geminiService.ts";
import { PromptTemplate } from "@langchain/core/prompts";
import { getUser, prompts } from "../../config.ts";

export const createChatNode = (geminiService: GeminiService) => {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      // only for LangSmith Studio - set defaults if not present
      if (!state.user) {
        state.user = getUser("ananeri")!;
        state.guardrailsEnabled = false;
      }

      const userPrompt = state.messages.at(-1)?.text!;
      const template = PromptTemplate.fromTemplate(prompts.system);
      // exemplo abaixo é mais inseguro!!
      // const systemPrompt = prompts.system
      //  .replace('{USER_ROLE}', state.user.role);
      //  .replace('{USER_NAME}', state.user.displayName);

      const systemPrompt = await template.format({
        USER_ROLE: state.user.role,
        USER_NAME: state.user.displayName,
      });

      const response = await geminiService.generate(systemPrompt, userPrompt);
      return {
        messages: [new AIMessage(response)],
      };
    } catch (error) {
      console.error("Chat node error:", error);
      return {
        messages: [
          new AIMessage(
            "I apologize, but I encountered an error processing your request. Please try again later.",
          ),
        ],
      };
    }
  };
};
