// CLS++ SDK — runnable quickstart.
//
// 1. Get a free API key at https://www.clsplusplus.com (1,000 requests/month).
// 2. Run:  CLS_API_KEY=cls_live_xxxxx node examples/quickstart.mjs
//
// This writes a couple of memories, recalls them semantically, builds an
// LLM-ready context string, then forgets one — the full Brain lifecycle.

import { Brain } from "clsplusplus"

if (!process.env.CLS_API_KEY) {
  console.error("Set CLS_API_KEY first — get a key at https://www.clsplusplus.com")
  process.exit(1)
}

const brain = new Brain("quickstart-demo")

await brain.learn("I work at Google as a senior engineer")
await brain.learn("I prefer Python over JavaScript")
console.log("learned 2 facts")

const job = await brain.ask("What's my job?")
console.log("ask('What's my job?') →", job)

const context = await brain.context("coding help")
console.log("context('coding help') →", context)

await brain.forget("I work at Google as a senior engineer")
console.log("forgot 1 fact — remaining:", await brain.count())
