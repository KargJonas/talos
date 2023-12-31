import { test } from "bun:test";
import { core_ready } from "../src/util";

test("core initialization", async () => await core_ready, 50);
