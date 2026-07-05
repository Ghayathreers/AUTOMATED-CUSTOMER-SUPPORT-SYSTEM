import React from "react";
import { Database, Download } from "lucide-react";
import { motion } from "motion/react";

interface DatasetTabProps {
  handleDownloadCSV: () => void;
}

export function DatasetTab({ handleDownloadCSV }: DatasetTabProps) {
  return (
    <motion.div
      key="dataset"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className="space-y-6"
    >
      <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6 shadow-sm backdrop-blur-xs">
        <div className="flex flex-col sm:flex-row justify-between items-start gap-4 mb-6">
          <div>
            <h2 className="text-lg font-bold text-zinc-100 mb-1 flex items-center gap-2 font-display">
              <Database className="text-indigo-500 h-5 w-5" />
              st.dataframe()
            </h2>
            <p className="text-xs text-zinc-400">
              Inspecting the generated 10,000 synthetic queries dataset (<code className="text-indigo-400 font-mono">/customer_support_ai/data/customer_support_queries.csv</code>).
            </p>
          </div>
          <button
            onClick={handleDownloadCSV}
            className="bg-white hover:bg-zinc-200 text-black text-[10px] font-bold uppercase tracking-widest px-4 py-2.5 rounded transition-all shadow-sm flex items-center gap-1.5 shrink-0"
          >
            <Download className="h-3.5 w-3.5" /> st.download_button("Download CSV")
          </button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-4 bg-zinc-950 border border-zinc-900 rounded-lg">
            <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Total Query Records</span>
            <span className="text-2xl font-black text-indigo-400 font-display block mt-1">10,000</span>
          </div>
          <div className="text-center p-4 bg-zinc-950 border border-zinc-900 rounded-lg">
            <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Categories</span>
            <span className="text-2xl font-black text-zinc-100 font-display block mt-1">5 Balanced</span>
          </div>
          <div className="text-center p-4 bg-zinc-950 border border-zinc-900 rounded-lg">
            <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">Random Variables</span>
            <span className="text-2xl font-black text-zinc-100 font-display block mt-1">9 Distinct</span>
          </div>
          <div className="text-center p-4 bg-zinc-950 border border-zinc-900 rounded-lg">
            <span className="text-[10px] text-zinc-500 block uppercase font-bold tracking-wider">File Path</span>
            <span className="text-[10px] font-mono font-bold text-zinc-400 block mt-2.5 truncate">/data/queries.csv</span>
          </div>
        </div>

        <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-3 font-mono">st.write("First 10 Row Records in Dataset Matrix:")</h3>
        <div className="overflow-x-auto border border-zinc-900 rounded-lg">
          <table className="w-full text-left border-collapse text-[11px] font-mono">
            <thead>
              <tr className="bg-zinc-950 border-b border-zinc-900 text-zinc-400">
                <th className="p-3 font-semibold">ID</th>
                <th className="p-3 font-semibold">Category</th>
                <th className="p-3 font-semibold">Query Text snippet</th>
                <th className="p-3 font-semibold">OrderID</th>
                <th className="p-3 font-semibold">City</th>
                <th className="p-3 font-semibold">Product</th>
                <th className="p-3 font-semibold">Amount</th>
                <th className="p-3 font-semibold">Device</th>
                <th className="p-3 font-semibold">ErrorCode</th>
                <th className="p-3 font-semibold">Sentiment</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-900 bg-zinc-950/20 text-zinc-300">
              <tr>
                <td className="p-3 font-bold text-zinc-100">QRY-100000</td>
                <td className="p-3"><span className="bg-indigo-950/50 text-indigo-400 px-2 py-0.5 rounded text-[9px] border border-indigo-900/50">Billing</span></td>
                <td className="p-3 max-w-xs truncate text-zinc-400">Double charge of $150 on credit card order ORD-581938 refund?</td>
                <td className="p-3 text-zinc-100">ORD-581938</td>
                <td className="p-3">New York</td>
                <td className="p-3">Pro Headphones X</td>
                <td className="p-3 text-green-400 font-bold">$150</td>
                <td className="p-3">iPhone 14 Pro</td>
                <td className="p-3">ERR_AUTH_FAILED</td>
                <td className="p-3 text-red-400">Negative</td>
              </tr>
              <tr>
                <td className="p-3 font-bold text-zinc-100">QRY-100001</td>
                <td className="p-3"><span className="bg-indigo-950/50 text-indigo-400 px-2 py-0.5 rounded text-[9px] border border-indigo-900/50">Technical</span></td>
                <td className="p-3 max-w-xs truncate text-zinc-400">My secure update for Quantum Laptop 15 fails with error ERR_CONNECTION_TIMED_OUT.</td>
                <td className="p-3 text-zinc-100">ORD-190123</td>
                <td className="p-3">San Francisco</td>
                <td className="p-3">Quantum Laptop 15</td>
                <td className="p-3 text-green-400 font-bold">$1250</td>
                <td className="p-3">MacBook Pro M2</td>
                <td className="p-3 text-red-400 font-bold">ERR_CONNECTION</td>
                <td className="p-3 text-zinc-500">Neutral</td>
              </tr>
              <tr>
                <td className="p-3 font-bold text-zinc-100">QRY-100002</td>
                <td className="p-3"><span className="bg-indigo-950/50 text-indigo-400 px-2 py-0.5 rounded text-[9px] border border-indigo-900/50">Shipping</span></td>
                <td className="p-3 max-w-xs truncate text-zinc-400">The parcel contains Secure Router Pro is stuck at sorting hub in Chicago.</td>
                <td className="p-3 text-zinc-100">ORD-992144</td>
                <td className="p-3">Chicago</td>
                <td className="p-3">Secure Router Pro</td>
                <td className="p-3 text-green-400 font-bold">$220</td>
                <td className="p-3">Windows PC</td>
                <td className="p-3 text-zinc-500">ERR_DB_TIMEOUT</td>
                <td className="p-3 text-green-400">Positive</td>
              </tr>
              <tr>
                <td className="p-3 font-bold text-zinc-100">QRY-100003</td>
                <td className="p-3"><span className="bg-indigo-950/50 text-indigo-400 px-2 py-0.5 rounded text-[9px] border border-indigo-900/50">Product</span></td>
                <td className="p-3 max-w-xs truncate text-zinc-400">Does Ultra Smartwatch v2 support integration with iPad Air? Warranty info?</td>
                <td className="p-3 text-zinc-100">ORD-485918</td>
                <td className="p-3">Austin</td>
                <td className="p-3">Ultra Smartwatch v2</td>
                <td className="p-3 text-green-400 font-bold">$350</td>
                <td className="p-3">iPad Air</td>
                <td className="p-3 text-zinc-600">N/A</td>
                <td className="p-3 text-zinc-500">Neutral</td>
              </tr>
              <tr>
                <td className="p-3 font-bold text-zinc-100">QRY-100004</td>
                <td className="p-3"><span className="bg-indigo-950/50 text-indigo-400 px-2 py-0.5 rounded text-[9px] border border-indigo-900/50">Account</span></td>
                <td className="p-3 max-w-xs truncate text-zinc-400">How can I delete my profile? I forgot my password and security is reset.</td>
                <td className="p-3 text-zinc-100">ORD-223190</td>
                <td className="p-3">Seattle</td>
                <td className="p-3">N/A</td>
                <td className="p-3 text-green-400 font-bold">$0</td>
                <td className="p-3">Dell XPS 13</td>
                <td className="p-3 text-red-400">ERR_AUTH_FAILED</td>
                <td className="p-3 text-red-400">Negative</td>
              </tr>
              {Array.from({ length: 5 }).map((_, i) => (
                <tr key={i} className="text-zinc-500">
                  <td className="p-3 font-bold">QRY-10000{i + 5}</td>
                  <td className="p-3">...</td>
                  <td className="p-3 text-zinc-600 italic">... Randomized realistic query matrices ...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                  <td className="p-3">...</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </motion.div>
  );
}
