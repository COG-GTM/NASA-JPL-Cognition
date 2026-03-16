import { scienceDataModule } from "./scienceData";
import { spacecraftControlModule } from "./spacecraftControl";
import { monteCarloModule } from "./monteCarlo";
import type { Module } from "../types";

export const modules: Module[] = [
  scienceDataModule,
  spacecraftControlModule,
  monteCarloModule,
];
