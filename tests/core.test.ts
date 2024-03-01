import { test } from "bun:test";
import { core_ready } from "../index";


test("core initialization", async () => await core_ready, 50);
