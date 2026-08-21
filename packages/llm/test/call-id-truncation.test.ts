import { describe, expect } from "bun:test"
import { Effect } from "effect"
import { LLM, Message, ToolCallPart } from "../src"
import * as OpenAIChat from "../src/protocols/openai-chat"
import * as OpenAIResponses from "../src/protocols/openai-responses"
import { Auth } from "../src/route"
import { LLMClient } from "../src/route"
import { it } from "./lib/effect"

const responsesModel = OpenAIResponses.route
  .with({ endpoint: { baseURL: "https://api.openai.test/v1/" }, auth: Auth.bearer("test") })
  .model({ id: "gpt-4.1-mini" })

const chatModel = OpenAIChat.route
  .with({ endpoint: { baseURL: "https://api.openai.test/v1/" }, auth: Auth.bearer("test") })
  .model({ id: "gpt-4o-mini" })

const longId = "x".repeat(80)
const expectedTruncated = longId.slice(0, 64)
const shortId = "call_1"

type ResponsesBody = OpenAIResponses.OpenAIResponsesBody
type ChatBody = OpenAIChat.OpenAIChatBody

describe("call_id truncation to 64 chars", () => {
  it.effect("OpenAI Responses lowerToolCall truncates call_id >64 to 64", () =>
    Effect.gen(function* () {
      const prepared = yield* LLMClient.prepare<ResponsesBody>(
        LLM.request({
          model: responsesModel,
          messages: [Message.assistant([ToolCallPart.make({ id: longId, name: "lookup", input: { query: "weather" } })])],
        }),
      )
      const call = prepared.body.input.find((item: any) => item.type === "function_call") as any
      expect(call).toBeDefined()
      expect(call.call_id).toBe(expectedTruncated)
      expect(call.call_id.length).toBe(64)
    }),
  )

  it.effect("OpenAI Responses lowerToolCall passes short IDs unchanged", () =>
    Effect.gen(function* () {
      const prepared = yield* LLMClient.prepare<ResponsesBody>(
        LLM.request({
          model: responsesModel,
          messages: [Message.assistant([ToolCallPart.make({ id: shortId, name: "lookup", input: { query: "weather" } })])],
        }),
      )
      const call = prepared.body.input.find((item: any) => item.type === "function_call") as any
      expect(call.call_id).toBe(shortId)
    }),
  )

  it.effect("OpenAI Responses lowerMessages truncates tool output call_id >64 to 64", () =>
    Effect.gen(function* () {
      const prepared = yield* LLMClient.prepare<ResponsesBody>(
        LLM.request({
          model: responsesModel,
          messages: [
            Message.assistant([ToolCallPart.make({ id: longId, name: "lookup", input: { query: "weather" } })]),
            Message.tool({ id: longId, name: "lookup", result: { forecast: "sunny" } }),
          ],
        }),
      )
      const output = prepared.body.input.find((item: any) => item.type === "function_call_output") as any
      expect(output).toBeDefined()
      expect(output.call_id).toBe(expectedTruncated)
      expect(output.call_id.length).toBe(64)
    }),
  )

  it.effect("OpenAI Responses lowerMessages passes short tool output IDs unchanged", () =>
    Effect.gen(function* () {
      const prepared = yield* LLMClient.prepare<ResponsesBody>(
        LLM.request({
          model: responsesModel,
          messages: [
            Message.assistant([ToolCallPart.make({ id: shortId, name: "lookup", input: { query: "weather" } })]),
            Message.tool({ id: shortId, name: "lookup", result: { forecast: "sunny" } }),
          ],
        }),
      )
      const output = prepared.body.input.find((item: any) => item.type === "function_call_output") as any
      expect(output.call_id).toBe(shortId)
    }),
  )

  it.effect("OpenAI Chat lowerToolMessages truncates tool_call_id >64 to 64", () =>
    Effect.gen(function* () {
      const prepared = yield* LLMClient.prepare<ChatBody>(
        LLM.request({
          model: chatModel,
          messages: [
            Message.assistant([ToolCallPart.make({ id: longId, name: "lookup", input: { query: "weather" } })]),
            Message.tool({ id: longId, name: "lookup", result: { forecast: "sunny" } }),
          ],
        }),
      )
      const toolMsg = prepared.body.messages.find((m: any) => m.role === "tool") as any
      expect(toolMsg).toBeDefined()
      expect(toolMsg.tool_call_id).toBe(expectedTruncated)
      expect(toolMsg.tool_call_id.length).toBe(64)
    }),
  )

  it.effect("OpenAI Chat lowerToolMessages passes short IDs unchanged", () =>
    Effect.gen(function* () {
      const prepared = yield* LLMClient.prepare<ChatBody>(
        LLM.request({
          model: chatModel,
          messages: [
            Message.assistant([ToolCallPart.make({ id: shortId, name: "lookup", input: { query: "weather" } })]),
            Message.tool({ id: shortId, name: "lookup", result: { forecast: "sunny" } }),
          ],
        }),
      )
      const toolMsg = prepared.body.messages.find((m: any) => m.role === "tool") as any
      expect(toolMsg.tool_call_id).toBe(shortId)
    }),
  )

  it.effect("OpenAI Chat handles content-type tool results with truncation", () =>
    Effect.gen(function* () {
      const prepared = yield* LLMClient.prepare<ChatBody>(
        LLM.request({
          model: chatModel,
          messages: [
            Message.assistant([ToolCallPart.make({ id: longId, name: "read", input: { path: "pixel.png" } })]),
            Message.tool({
              id: longId,
              name: "read",
              result: {
                type: "content",
                value: [{ type: "text", text: "Image read successfully" }],
              },
            }),
          ],
        }),
      )
      const toolMsg = prepared.body.messages.find((m: any) => m.role === "tool") as any
      expect(toolMsg.tool_call_id).toBe(expectedTruncated)
    }),
  )
})
// TDD verification for call_id truncation - ensures slice(0,64) for >64 ids
