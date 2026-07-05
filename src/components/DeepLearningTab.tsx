import React from "react";
import { Activity, Sliders, CheckCircle, RefreshCw } from "lucide-react";
import { motion } from "motion/react";

interface DeepLearningTabProps {
  dlPipelineResults: any;
}

export function DeepLearningTab({ dlPipelineResults }: DeepLearningTabProps) {
  return (
    <motion.div
      key="dl"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className="space-y-6"
    >
      <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6 shadow-sm backdrop-blur-xs">
        <h2 className="text-lg font-bold text-zinc-100 mb-1 flex items-center gap-2 font-display">
          <Activity className="text-indigo-500 h-5 w-5" />
          st.line_chart("Neural Network Learning Curves")
        </h2>
        <p className="text-xs text-zinc-400 mb-6">
          Multi-Layer Perceptron (MLP), LSTM, and Bidirectional LSTM (BiLSTM) model testing sequences. Sequences are padded to size 20 and passed through a 16-dimensional continuous Embedding lookup.
        </p>

        {dlPipelineResults ? (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Deep Architecture Champion</span>
                <span className="text-sm font-black text-indigo-400 block mt-1 font-mono">Bidirectional LSTM</span>
              </div>
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">BiLSTM Accuracy Score</span>
                <span className="text-lg font-black text-zinc-100 block mt-1 font-display">{(dlPipelineResults.bilstmMetrics.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Gains over SVM baseline</span>
                <span className="text-lg font-black text-green-400 block mt-1 font-display">+{(dlPipelineResults.comparisonWithML.improvement * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Early Stopping Epoch</span>
                <span className="text-sm font-bold text-amber-500 block mt-2 font-mono">Epoch {dlPipelineResults.earlyStoppedEpoch} (Patience 3)</span>
              </div>
            </div>

            {/* Learning Curves Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Epochs Accuracy lines SVG */}
              <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-3 font-mono">st.line_chart("Training Validation Accuracy Curves")</span>
                <div className="h-60 bg-zinc-900/20 border border-zinc-900 rounded-lg p-3 relative flex items-end">
                  <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {/* Horizontal guidelines */}
                    <line x1="0" y1="20" x2="100" y2="20" stroke="#1f1f23" strokeWidth="1" />
                    <line x1="0" y1="40" x2="100" y2="40" stroke="#1f1f23" strokeWidth="1" />
                    <line x1="0" y1="60" x2="100" y2="60" stroke="#1f1f23" strokeWidth="1" />
                    <line x1="0" y1="80" x2="100" y2="80" stroke="#1f1f23" strokeWidth="1" />

                    {/* LSTM Curve (Purple) */}
                    <path 
                      d={`M 0 54 ${dlPipelineResults.lstmAccCurve.map((acc: number, i: number) => `L ${(i / (dlPipelineResults.lstmAccCurve.length - 1)) * 100} ${100 - (acc * 100)}`).join(" ")}`}
                      fill="none"
                      stroke="#6366f1"
                      strokeWidth="2.5"
                    />

                    {/* MLP Curve (Red-Orange) */}
                    <path 
                      d={`M 0 48 ${dlPipelineResults.mlpAccCurve.map((acc: number, i: number) => `L ${(i / (dlPipelineResults.mlpAccCurve.length - 1)) * 100} ${100 - (acc * 100)}`).join(" ")}`}
                      fill="none"
                      stroke="#71717a"
                      strokeWidth="2"
                    />
                  </svg>
                  <div className="absolute top-2 right-2 flex gap-3 text-[9px] font-bold font-mono bg-zinc-950 p-1.5 rounded border border-zinc-800">
                    <span className="flex items-center gap-1"><span className="h-1.5 w-3 bg-indigo-500 rounded" />LSTM (Validation)</span>
                    <span className="flex items-center gap-1"><span className="h-1.5 w-3 bg-zinc-500 rounded" />MLP (Validation)</span>
                  </div>
                  <span className="absolute bottom-1 left-2 text-[8px] font-mono text-zinc-600">Epoch 1</span>
                  <span className="absolute bottom-1 right-2 text-[8px] font-mono text-zinc-600">Epoch 10</span>
                </div>
              </div>

              {/* Epochs Loss lines SVG */}
              <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-3 font-mono">st.line_chart("Categorical Cross-Entropy Loss curves")</span>
                <div className="h-60 bg-zinc-900/20 border border-zinc-900 rounded-lg p-3 relative flex items-end">
                  <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {/* Guidelines */}
                    <line x1="0" y1="20" x2="100" y2="20" stroke="#1f1f23" strokeWidth="1" />
                    <line x1="0" y1="40" x2="100" y2="40" stroke="#1f1f23" strokeWidth="1" />
                    <line x1="0" y1="60" x2="100" y2="60" stroke="#1f1f23" strokeWidth="1" />
                    <line x1="0" y1="80" x2="100" y2="80" stroke="#1f1f23" strokeWidth="1" />

                    {/* LSTM Loss Curve (Purple) */}
                    <path 
                      d={`M 0 95 ${dlPipelineResults.lstmLossCurve.map((loss: number, i: number) => `L ${(i / (dlPipelineResults.lstmLossCurve.length - 1)) * 100} ${100 - (loss * 70)}`).join(" ")}`}
                      fill="none"
                      stroke="#6366f1"
                      strokeWidth="2.5"
                    />

                    {/* MLP Loss Curve (Red-Orange) */}
                    <path 
                      d={`M 0 90 ${dlPipelineResults.mlpLossCurve.map((loss: number, i: number) => `L ${(i / (dlPipelineResults.mlpLossCurve.length - 1)) * 100} ${100 - (loss * 70)}`).join(" ")}`}
                      fill="none"
                      stroke="#71717a"
                      strokeWidth="2"
                    />
                  </svg>
                  <span className="absolute bottom-1 left-2 text-[8px] font-mono text-zinc-600">Epoch 1</span>
                  <span className="absolute bottom-1 right-2 text-[8px] font-mono text-zinc-600">Epoch 10</span>
                </div>
              </div>
            </div>

            {/* Performance vs Traditional ML */}
            <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-3 font-mono">st.dataframe("Deep Learning vs. Traditional ML accuracy comparison matrix")</span>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-zinc-900/30 border border-zinc-900 p-4 rounded-lg flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Traditional ML Baseline</span>
                    <span className="text-sm font-bold text-zinc-400 block mt-0.5">{(dlPipelineResults.comparisonWithML.bestMLAccuracy * 100).toFixed(1)}%</span>
                  </div>
                  <Sliders className="h-5 w-5 text-zinc-600" />
                </div>
                <div className="bg-zinc-900/30 border border-zinc-900 p-4 rounded-lg flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Deep Neural BiLSTM</span>
                    <span className="text-sm font-bold text-indigo-400 block mt-0.5">{(dlPipelineResults.comparisonWithML.bestDLAccuracy * 100).toFixed(1)}%</span>
                  </div>
                  <CheckCircle className="h-5 w-5 text-indigo-500" />
                </div>
                <div className="bg-zinc-900/30 border border-zinc-900 p-4 rounded-lg flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Net Accuracy Improvement</span>
                    <span className="text-sm font-bold text-green-400 block mt-0.5">+{(dlPipelineResults.comparisonWithML.improvement * 100).toFixed(1)}%</span>
                  </div>
                  <span className="text-[9px] bg-green-500/10 border border-green-500/20 text-green-400 font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">Gain</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center p-10 bg-zinc-950 border border-zinc-900 rounded-lg">
            <RefreshCw className="h-6 w-6 animate-spin mx-auto text-indigo-500 mb-2" />
            <p className="text-xs text-zinc-400">Loading Deep Learning learning curves. Verify FastAPI backend diagnostics.</p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
