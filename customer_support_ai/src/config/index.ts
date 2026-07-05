/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import * as path from "path";

export const PROJECT_ROOT = path.join(process.cwd(), "customer_support_ai");
export const DATA_DIR = path.join(PROJECT_ROOT, "data");
export const MODELS_DIR = path.join(PROJECT_ROOT, "models");
export const OUTPUTS_DIR = path.join(PROJECT_ROOT, "outputs");

export const CONFIG = {
  projectRoot: PROJECT_ROOT,
  dataDir: DATA_DIR,
  modelsDir: MODELS_DIR,
  outputsDir: OUTPUTS_DIR,
  maxMLFeatures: 120,
  maxDLSequenceLength: 20,
  maxDLFeatures: 250,
  trainRatio: 0.8,
  crossValidationFolds: 3
};
