import React from "react";
import { Sliders, RefreshCw } from "lucide-react";
import { motion } from "motion/react";

interface MachineLearningTabProps {
  mlPipelineResults: any;
}

export function MachineLearningTab({ mlPipelineResults }: MachineLearningTabProps) {
  return (
    <motion.div
      key="ml"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className="space-y-6"
    >
      <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6 shadow-sm backdrop-blur-xs">
        <h2 className="text-lg font-bold text-zinc-100 mb-1 flex items-center gap-2 font-display">
          <Sliders className="text-indigo-500 h-5 w-5" />
          st.pyplot("Traditional ML Performance Report")
        </h2>
        <p className="text-xs text-zinc-400 mb-6">
          Performance metrics evaluated over 80/20 train/test split of 10,000 queries. Features are vectorized using a vocabulary of top TF-IDF unigrams.
        </p>

        {mlPipelineResults ? (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Best Classifier Model</span>
                <span className="text-sm font-black text-indigo-400 block mt-1 font-mono">{mlPipelineResults.bestModelName}</span>
              </div>
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Logistic Regression Accuracy</span>
                <span className="text-lg font-black text-zinc-100 block mt-1 font-display">{(mlPipelineResults.logisticRegression.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Random Forest Accuracy</span>
                <span className="text-lg font-black text-zinc-100 block mt-1 font-display">{(mlPipelineResults.randomForest.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-xl text-center">
                <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">SVM Accuracy</span>
                <span className="text-lg font-black text-zinc-100 block mt-1 font-display">{(mlPipelineResults.svm.accuracy * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Visual Bar Chart for Model Comparision */}
            <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-4 font-mono">st.bar_chart("Accuracy Metric Comparisons:")</span>
              <div className="h-64 flex items-end justify-around border-b border-zinc-800 pb-2 px-6 sm:px-10 relative">
                {/* Logistic Regression Bar */}
                <div className="flex flex-col items-center w-20 sm:w-24 z-10">
                  <div 
                    style={{ height: `${mlPipelineResults.logisticRegression.accuracy * 180}px` }}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-t flex items-center justify-center text-zinc-300 text-xs font-bold"
                  >
                    {(mlPipelineResults.logisticRegression.accuracy * 100).toFixed(1)}%
                  </div>
                  <span className="text-[10px] font-bold text-zinc-500 mt-2 font-mono">Logistic Reg.</span>
                </div>

                {/* Random Forest Bar */}
                <div className="flex flex-col items-center w-20 sm:w-24 z-10">
                  <div 
                    style={{ height: `${mlPipelineResults.randomForest.accuracy * 180}px` }}
                    className="w-full bg-zinc-700 border border-zinc-600 rounded-t flex items-center justify-center text-zinc-100 text-xs font-bold"
                  >
                    {(mlPipelineResults.randomForest.accuracy * 100).toFixed(1)}%
                  </div>
                  <span className="text-[10px] font-bold text-zinc-500 mt-2 font-mono">Random Forest</span>
                </div>

                {/* SVM Bar */}
                <div className="flex flex-col items-center w-20 sm:w-24 z-10">
                  <div 
                    style={{ height: `${mlPipelineResults.svm.accuracy * 180}px` }}
                    className="w-full bg-indigo-600 border border-indigo-500 rounded-t flex items-center justify-center text-white text-xs font-bold shadow-[0_0_12px_rgba(99,102,241,0.4)]"
                  >
                    {(mlPipelineResults.svm.accuracy * 100).toFixed(1)}%
                  </div>
                  <span className="text-[10px] font-bold text-indigo-400 mt-2 font-mono">SVM SGD</span>
                </div>

                {/* Horizontal Grid lines */}
                <div className="absolute left-0 right-0 border-b border-dashed border-zinc-900 top-[33%]" />
                <div className="absolute left-0 right-0 border-b border-dashed border-zinc-900 top-[66%]" />
              </div>
            </div>

            {/* Feature Importance & Confusion Matrix */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-3 font-mono">st.write("Top 10 Feature Importance (TF-IDF Coefficients)")</span>
                <div className="space-y-2">
                  {mlPipelineResults.featureImportance.slice(0, 10).map((feat: any, idx: number) => (
                    <div key={idx} className="flex items-center justify-between text-xs bg-zinc-900/30 border border-zinc-900 p-2.5 rounded-lg">
                      <span className="font-mono text-zinc-300">"{feat.feature}"</span>
                      <div className="flex items-center gap-3">
                        <div className="w-20 sm:w-24 bg-zinc-900 h-2 rounded-full overflow-hidden">
                          <div 
                            style={{ width: `${Math.min(feat.importance * 40, 100)}%` }} 
                            className="bg-indigo-500 h-full" 
                          />
                        </div>
                        <span className="font-bold text-zinc-400 font-mono">{(feat.importance).toFixed(4)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Confusion Matrix Visual */}
              <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-3 font-mono">st.write("Confusion Matrix Heatmap (Act vs Pred)")</span>
                <div className="grid grid-cols-6 gap-1 text-center font-mono text-[9px] bg-zinc-900/10 p-3 border border-zinc-900 rounded-lg">
                  <div className="font-bold text-zinc-500 self-center">Act\Pred</div>
                  {mlPipelineResults.confusionMatrix.labels.map((l: string) => (
                    <div key={l} className="font-bold text-zinc-400 truncate py-1">{l[0]}..</div>
                  ))}

                  {mlPipelineResults.confusionMatrix.labels.map((rowLabel: string, rIdx: number) => (
                    <React.Fragment key={rowLabel}>
                      <div className="font-bold text-zinc-400 self-center text-left truncate py-1">{rowLabel}</div>
                      {mlPipelineResults.confusionMatrix.matrix[rIdx].map((cell: number, cIdx: number) => {
                        const isDiagonal = rIdx === cIdx;
                        return (
                          <div 
                            key={cIdx} 
                            className={`p-2 rounded font-bold flex items-center justify-center border border-zinc-950 ${
                              isDiagonal ? "bg-indigo-600 text-white shadow-[0_0_8px_rgba(99,102,241,0.3)]" : "bg-zinc-900/40 text-zinc-600"
                            }`}
                          >
                            {cell}
                          </div>
                        );
                      })}
                    </React.Fragment>
                  ))}
                </div>
                <p className="text-[9px] text-zinc-500 font-mono mt-3 text-center">Abbreviations: B: Billing, T: Technical, S: Shipping, P: Product, A: Account</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center p-10 bg-zinc-950 border border-zinc-900 rounded-lg">
            <RefreshCw className="h-6 w-6 animate-spin mx-auto text-indigo-500 mb-2" />
            <p className="text-xs text-zinc-400">Loading ML analysis results. Verify FastAPI server synchronization.</p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
