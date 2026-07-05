import React from "react";
import { Workflow } from "lucide-react";
import { motion } from "motion/react";

export function LldTab() {
  return (
    <motion.div
      key="lld"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className="space-y-6"
    >
      <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6 shadow-sm backdrop-blur-xs">
        <h2 className="text-lg font-bold text-zinc-100 mb-1 flex items-center gap-2 font-display">
          <Workflow className="text-indigo-500 h-5 w-5" />
          st.write("Level 5: Low-Level Design UML Schematics")
        </h2>
        <p className="text-xs text-zinc-400 mb-6">
          A comprehensive overview of component interactions, class contracts, data flows, and technical infrastructure schemas.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Class Diagram */}
          <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-3 font-mono">UML Structural Class Diagram</span>
            <div className="bg-zinc-950 text-zinc-400 font-mono text-[9px] p-4 rounded-lg leading-relaxed max-h-96 overflow-y-auto border border-zinc-900">
              <div className="text-indigo-400 font-bold mb-1">class SupportQuery</div>
              <div className="pl-4 mb-2">
                + id: string<br />
                + query: string<br />
                + category: QueryCategory<br />
                + orderId: string<br />
                + customerName: string<br />
              </div>

              <div className="text-indigo-400 font-bold mb-1">class MLPipeline</div>
              <div className="pl-4 mb-2">
                + fit(X: number[][], y: string[]): void<br />
                + predict(x: number[]): string<br />
                + evaluateClassifier(): ModelMetrics<br />
              </div>

              <div className="text-indigo-400 font-bold mb-1">class Tokenizer</div>
              <div className="pl-4 mb-2">
                + fit(texts: string[]): void<br />
                + textsToSequences(texts: string[]): number[][]<br />
              </div>

              <div className="text-indigo-400 font-bold mb-1">class AgenticAI</div>
              <div className="pl-4">
                - memory: AgenticMemory<br />
                + runAgenticLoop(queryText: string): Promise&lt;AgenticResult&gt;<br />
                - checkSafety(): boolean<br />
                - checkHumanEscalationLimits(): boolean<br />
              </div>
            </div>
          </div>

          {/* Sequence Flow */}
          <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-3 font-mono">Sequence Flow & Data pipelines</span>
            <div className="bg-zinc-950 text-zinc-400 font-mono text-[9px] p-4 rounded-lg leading-relaxed max-h-96 overflow-y-auto border border-zinc-900">
              <div className="text-green-400 mb-3 font-bold">[User Query Input]</div>
              <div className="pl-4 border-l border-zinc-800 space-y-3">
                <div>➔ <span className="text-zinc-200">Step 1: SafetyGuard.check()</span></div>
                <div className="pl-4 text-zinc-500 font-sans italic">// Inspects toxic inputs & prompt injections</div>

                <div>➔ <span className="text-zinc-200">Step 2: NLP.runPipeline()</span></div>
                <div className="pl-4 text-zinc-500 font-sans italic">// Cleaning, stopword removal, lemmatizing, regex, sentiment</div>

                <div>➔ <span className="text-zinc-200">Step 3: Classifier.predict()</span></div>
                <div className="pl-4 text-zinc-500 font-sans italic">// Run Logistic Regression / BiLSTM category check</div>

                <div>➔ <span className="text-zinc-200">Step 4: AgenticAI.checkEscalationLimits()</span></div>
                <div className="pl-4 text-zinc-500 font-sans italic">// If refund &gt; $500, trigger HumanEscalation</div>

                <div>➔ <span className="text-zinc-200">Step 5: SLM & GenAI Draft Resolution</span></div>
                <div className="pl-4 text-zinc-500 font-sans italic">// Summarizes, builds FAQs, generates personalized response</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
