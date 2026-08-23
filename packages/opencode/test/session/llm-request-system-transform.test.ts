import { describe, expect } from "bun:test"
import { Effect } from "effect"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Npm } from "@opencode-ai/core/npm"
import path from "path"
import { pathToFileURL } from "url"
import { Account } from "../../src/account/account"
import { Auth } from "../../src/auth"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { Plugin } from "../../src/plugin/index"
import { prepare } from "../../src/session/llm/request"
import { TestInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { AccountTest } from "../fake/account"
import { AuthTest } from "../fake/auth"
import { NpmTest } from "../fake/npm"

const it = testEffect(
  AppNodeBuilder.build(LayerNode.group([Plugin.node, CrossSpawnSpawner.node]), [
    [Auth.node, AuthTest.empty],
    [Account.node, AccountTest.empty],
    [Npm.node, NpmTest.noop],
    [RuntimeFlags.node, RuntimeFlags.layer({ disableDefaultPlugins: true })],
  ]),
)

function withProject<A, E, R>(source: string, self: Effect.Effect<A, E, R>) {
  return Effect.gen(function* () {
    const test = yield* TestInstance
    const file = path.join(test.directory, "plugin.ts")
    yield* Effect.all(
      [
        Effect.promise(() => Bun.write(file, source)),
        Effect.promise(() =>
          Bun.write(
            path.join(test.directory, "opencode.json"),
            JSON.stringify(
              {
                $schema: "https://opencode.ai/config.json",
                plugin: [pathToFileURL(file).href],
              },
              null,
              2,
            ),
          ),
        ),
      ],
      { discard: true, concurrency: 2 },
    )
    return yield* self
  })
}

describe("llm request prep", () => {
  // Regression guard: the Aug-15 merge dropped upstream's request-side
  // experimental.chat.system.transform firing, silently disabling
  // magic-context guidance injection and aft-opencode hints.
  it.instance("fires experimental.chat.system.transform with the assembled system", () =>
    withProject(
      [
        "export default async () => ({",
        '  "experimental.chat.system.transform": async (_input, output) => {',
        '    output.system.push("injected")',
        "  },",
        "})",
        "",
      ].join("\n"),
      Effect.gen(function* () {
        const plugin = yield* Plugin.Service
        const prepared = yield* prepare({
          user: {
            id: "usr_test",
            sessionID: "ses_test",
            model: { providerID: "test", modelID: "m1" },
          },
          sessionID: "ses_test",
          model: {
            id: "test/m1",
            providerID: "test",
            api: { id: "m1", npm: undefined },
            capabilities: { temperature: false },
            options: {},
            limit: { context: 1000, output: 100 },
          },
          agent: { name: "build", permission: [] },
          system: ["BASE"],
          messages: [],
          small: false,
          tools: {},
          provider: { id: "test", options: {} },
          auth: undefined,
          plugin,
          flags: {},
          isWorkflow: false,
        } as any)
        expect(prepared.system.some((s) => s.startsWith("injected"))).toBe(true)
      }),
    ),
  )
})
