import React from "react";
import { Compass } from "lucide-react";
import { motion } from "motion/react";

interface SlmTabProps {
  slmResult: any;
}

export function SlmTab({ slmResult }: SlmTabProps) {
  return (
    <motion.div
      key="slm"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className="space-y-6"
    >
      <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6 shadow-sm backdrop-blur-xs">
        <h2 className="text-lg font-bold text-zinc-100 mb-1 flex items-center gap-2 font-display">
          <Compass className="text-indigo-500 h-5 w-5" />
          st.write("Level 4: Small Language Model summarizer & evaluations")
        </h2>
        <p className="text-xs text-zinc-400 mb-6">
          A small local summarizer generating multi-sentence briefs, task checklists, and drafts. The quality is compared dynamically to a ground-truth summary using standard ROUGE overlap.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 bg-zinc-950 border border-zinc-900 p-5 rounded-xl space-y-4">
            <div className="bg-zinc-900/30 border border-zinc-900 p-4 rounded-lg">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-1.5 font-mono">Generated Summary:</span>
              <p className="text-xs text-zinc-200 font-medium leading-relaxed italic">
                {slmResult ? `"${slmResult.summary}"` : '"Run agent query in playground first to generate summaries."'}
              </p>
            </div>

            <div className="bg-zinc-900/30 border border-zinc-900 p-4 rounded-lg">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-1.5 font-mono">ROUGE Ground-Truth Reference Statement:</span>
              <p className="text-xs text-zinc-400 font-medium leading-relaxed italic">
                "The customer is requesting support with a related issue or transaction details."
              </p>
            </div>
          </div>

          <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-4 font-mono">ROUGE Overlap Calculations:</span>
            {slmResult ? (
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs font-mono font-bold text-zinc-400 mb-1.5">
                    <span>ROUGE-1 (Unigram Recall)</span>
                    <span className="text-zinc-200">{(slmResult.evaluation.rouge1 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                    <div style={{ width: `${slmResult.evaluation.rouge1 * 100}%` }} className="bg-indigo-500 h-full" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono font-bold text-zinc-400 mb-1.5">
                    <span>ROUGE-2 (Bigram Order Recall)</span>
                    <span className="text-zinc-200">{(slmResult.evaluation.rouge2 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                    <div style={{ width: `${slmResult.evaluation.rouge2 * 100}%` }} className="bg-zinc-500 h-full" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono font-bold text-zinc-400 mb-1.5">
                    <span>ROUGE-L (LCS Sequence Recall)</span>
                    <span className="text-zinc-200">{(slmResult.evaluation.rougeL * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                    <div style={{ width: `${slmResult.evaluation.rougeL * 100}%` }} className="bg-zinc-100 h-full" />
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-zinc-500 italic mt-10 text-center font-mono">Inference a query in st.playground first to compute overlaps.</p>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
