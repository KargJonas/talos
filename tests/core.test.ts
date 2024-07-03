import { core_ready } from "../index";
import { test } from "bun:test";


test("core initialization", async () => await core_ready, 50);
