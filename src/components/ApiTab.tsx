import React from "react";
import { FileCode } from "lucide-react";
import { motion } from "motion/react";

export function ApiTab() {
  return (
    <motion.div
      key="api"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className="space-y-6"
    >
      <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6 shadow-sm backdrop-blur-xs">
        <h2 className="text-lg font-bold text-zinc-100 mb-1 flex items-center gap-2 font-display">
          <FileCode className="text-indigo-500 h-5 w-5" />
          st.write("FastAPI-like Express API Contracts")
        </h2>
        <p className="text-xs text-zinc-400 mb-6">
          A comprehensive reference catalog of the runnable API endpoints. These contracts follow standard FastAPI architectures.
        </p>

        <div className="space-y-4">
          <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-lg">
            <div className="flex items-center gap-3 mb-2">
              <span className="bg-green-500/10 text-green-400 border border-green-500/20 font-mono text-[9px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">GET</span>
              <span className="font-mono text-xs font-bold text-zinc-200">/health</span>
            </div>
            <p className="text-xs text-zinc-500 mb-2">Service status and verification.</p>
            <pre className="bg-zinc-950 text-zinc-400 font-mono text-[9px] p-3 rounded-lg border border-zinc-900 overflow-x-auto">
{`{
  "status": "healthy",
  "service": "Automated Customer Support and Service Resolution AI",
  "pipelineState": {
    "datasetGenerated": true,
    "mlModelTrained": true,
    "dlModelTrained": true
  }
}`}
            </pre>
          </div>

          <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-lg">
            <div className="flex items-center gap-3 mb-2">
              <span className="bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono text-[9px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">POST</span>
              <span className="font-mono text-xs font-bold text-zinc-200">/predict</span>
            </div>
            <p className="text-xs text-zinc-500 mb-2">Category classification predictions over TF-IDF matrices.</p>
            <pre className="bg-zinc-950 text-zinc-400 font-mono text-[9px] p-3 rounded-lg border border-zinc-900 overflow-x-auto">
{`{
  "query": "Double charge of $150",
  "predictedCategory": "Billing",
  "confidenceScores": {
    "Billing": 0.892,
    "Technical": 0.024
  }
}`}
            </pre>
          </div>

          <div className="bg-zinc-950 border border-zinc-900 p-4 rounded-lg">
            <div className="flex items-center gap-3 mb-2">
              <span className="bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono text-[9px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">POST</span>
              <span className="font-mono text-xs font-bold text-zinc-200">/agent</span>
            </div>
            <p className="text-xs text-zinc-500 mb-2">Initiates cognitive agentic resolution cascades and logs safety sweeps.</p>
            <pre className="bg-zinc-950 text-zinc-400 font-mono text-[9px] p-3 rounded-lg border border-zinc-900 overflow-x-auto">
{`{
  "perception": {
    "detectedCategory": "Billing",
    "detectedSentiment": "Negative",
    "detectedUrgency": "High"
  },
  "reasoning": "Transaction logs evaluated. Value safe to authorize.",
  "safetyPassed": true,
  "humanEscalationRequired": false,
  "finalResolution": "Reply draft generated successfully."
}`}
            </pre>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
