import { useState } from "react";
import {
  Satellite,
  Rocket,
  BarChart3,
  ChevronRight,
  ChevronDown,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  Zap,
  Github,
  Terminal,
  BookOpen,
} from "lucide-react";
import { modules } from "./data";
import type { Module, UseCase } from "./types";
import "./App.css";

const iconMap: Record<string, React.ReactNode> = {
  satellite: <Satellite size={24} />,
  rocket: <Rocket size={24} />,
  chart: <BarChart3 size={24} />,
};

const iconMapLg: Record<string, React.ReactNode> = {
  satellite: <Satellite size={40} />,
  rocket: <Rocket size={40} />,
  chart: <BarChart3 size={40} />,
};

function CodeBlock({ code, variant }: { code: string; variant: "before" | "after" }) {
  return (
    <div className="relative">
      <div
        className={`absolute top-0 right-0 px-2 py-1 text-xs font-semibold rounded-bl-lg ${
          variant === "before"
            ? "bg-red-900/50 text-red-300"
            : "bg-green-900/50 text-green-300"
        }`}
      >
        {variant === "before" ? "BEFORE" : "AFTER"}
      </div>
      <pre
        className={`text-sm leading-relaxed overflow-x-auto p-4 rounded-lg border ${
          variant === "before"
            ? "bg-red-950/20 border-red-900/30"
            : "bg-green-950/20 border-green-900/30"
        }`}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

function MetricBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center px-4 py-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
      <span className="text-lg font-bold text-white">{value}</span>
      <span className="text-xs text-slate-400 mt-1">{label}</span>
    </div>
  );
}

function UseCaseCard({
  useCase,
  moduleColor,
  isExpanded,
  onToggle,
}: {
  useCase: UseCase;
  moduleColor: string;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="border border-slate-700/50 rounded-xl overflow-hidden bg-slate-900/50 transition-all duration-300 hover:border-slate-600/50">
      <button
        onClick={onToggle}
        className="w-full px-6 py-5 flex items-start gap-4 text-left hover:bg-slate-800/30 transition-colors cursor-pointer"
      >
        <div
          className="mt-1 flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${moduleColor}20` }}
        >
          {isExpanded ? (
            <ChevronDown size={18} style={{ color: moduleColor }} />
          ) : (
            <ChevronRight size={18} style={{ color: moduleColor }} />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-white">{useCase.title}</h3>
          <p className="text-sm text-slate-400 mt-1 line-clamp-2">
            {useCase.situation}
          </p>
          <div className="flex items-center gap-2 mt-2">
            <a
              href={useCase.nasaRepoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              <Github size={12} />
              {useCase.nasaRepo}
              <ExternalLink size={10} />
            </a>
          </div>
        </div>
        <div className="flex gap-2 flex-shrink-0">
          {useCase.metrics.map((m) => (
            <MetricBadge key={m.label} label={m.label} value={m.value} />
          ))}
        </div>
      </button>

      {isExpanded && (
        <div className="px-6 pb-6 space-y-6 border-t border-slate-700/50 pt-6">
          {/* Situation */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded-full bg-amber-900/30 flex items-center justify-center">
                <AlertTriangle size={14} className="text-amber-400" />
              </div>
              <h4 className="text-sm font-semibold text-amber-400 uppercase tracking-wider">
                The Situation
              </h4>
            </div>
            <p className="text-slate-300 text-sm leading-relaxed pl-8">
              {useCase.situation}
            </p>
          </div>

          {/* Before */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded-full bg-red-900/30 flex items-center justify-center">
                <AlertTriangle size={14} className="text-red-400" />
              </div>
              <h4 className="text-sm font-semibold text-red-400 uppercase tracking-wider">
                Before (Manual Workflow)
              </h4>
            </div>
            <p className="text-slate-300 text-sm leading-relaxed pl-8 mb-3">
              {useCase.before.description}
            </p>
            <div className="pl-8">
              <CodeBlock code={useCase.before.code} variant="before" />
            </div>
            <div className="pl-8 mt-3">
              <p className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-2">
                Pain Points
              </p>
              <ul className="space-y-1">
                {useCase.before.painPoints.map((pp, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-slate-400"
                  >
                    <span className="text-red-500 mt-0.5 flex-shrink-0">
                      &times;
                    </span>
                    {pp}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* What Devin Did */}
          <div
            className="rounded-lg p-4 border"
            style={{
              backgroundColor: `${moduleColor}08`,
              borderColor: `${moduleColor}30`,
            }}
          >
            <div className="flex items-center gap-2 mb-3">
              <div
                className="w-6 h-6 rounded-full flex items-center justify-center"
                style={{ backgroundColor: `${moduleColor}20` }}
              >
                <Zap size={14} style={{ color: moduleColor }} />
              </div>
              <h4
                className="text-sm font-semibold uppercase tracking-wider"
                style={{ color: moduleColor }}
              >
                What Devin Built
              </h4>
            </div>
            <p className="text-slate-300 text-sm leading-relaxed pl-8">
              {useCase.devinAction}
            </p>
          </div>

          {/* After */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded-full bg-green-900/30 flex items-center justify-center">
                <CheckCircle2 size={14} className="text-green-400" />
              </div>
              <h4 className="text-sm font-semibold text-green-400 uppercase tracking-wider">
                After (Devin-Enhanced)
              </h4>
            </div>
            <p className="text-slate-300 text-sm leading-relaxed pl-8 mb-3">
              {useCase.after.description}
            </p>
            <div className="pl-8">
              <CodeBlock code={useCase.after.code} variant="after" />
            </div>
            <div className="pl-8 mt-3">
              <p className="text-xs font-semibold text-green-400 uppercase tracking-wider mb-2">
                Improvements
              </p>
              <ul className="space-y-1">
                {useCase.after.improvements.map((imp, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-slate-400"
                  >
                    <CheckCircle2
                      size={14}
                      className="text-green-500 mt-0.5 flex-shrink-0"
                    />
                    {imp}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Value */}
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <ArrowRight size={16} className="text-blue-400" />
              <h4 className="text-sm font-semibold text-blue-400 uppercase tracking-wider">
                Value to NASA-JPL
              </h4>
            </div>
            <p className="text-slate-300 text-sm leading-relaxed pl-6">
              {useCase.value}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function ModuleSection({
  module,
  isActive,
}: {
  module: Module;
  isActive: boolean;
}) {
  const [expandedUseCase, setExpandedUseCase] = useState<string | null>(null);

  if (!isActive) return null;

  return (
    <div className="space-y-6">
      {/* Module Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <div
            className="w-14 h-14 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: `${module.color}15`, color: module.color }}
          >
            {iconMapLg[module.icon]}
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">{module.title}</h2>
            <p className="text-slate-400">{module.subtitle}</p>
          </div>
        </div>
        <p className="text-slate-300 text-sm leading-relaxed max-w-4xl">
          {module.description}
        </p>
        <div className="mt-4 flex items-center gap-2 text-sm text-slate-400">
          <BookOpen size={14} />
          <span>
            {module.useCases.length} use cases — click each to expand the full
            before/after analysis
          </span>
        </div>
      </div>

      {/* Use Cases */}
      <div className="space-y-4">
        {module.useCases.map((uc) => (
          <UseCaseCard
            key={uc.id}
            useCase={uc}
            moduleColor={module.color}
            isExpanded={expandedUseCase === uc.id}
            onToggle={() =>
              setExpandedUseCase(expandedUseCase === uc.id ? null : uc.id)
            }
          />
        ))}
      </div>
    </div>
  );
}

function App() {
  const [activeModule, setActiveModule] = useState(modules[0].id);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <Rocket size={22} className="text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">
                  NASA-JPL Science & Engineering Toolkit
                </h1>
                <p className="text-xs text-slate-400">
                  AI-Accelerated Workflows for Space Science & Mission Engineering
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/COG-GTM/NASA-JPL-Cognition"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors text-sm"
              >
                <Github size={16} />
                View Source
              </a>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-sm">
                <Terminal size={16} />
                <span>
                  <code className="text-xs">pip install -e . &amp;&amp; cd web &amp;&amp; npm run dev</code>
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-900/30 text-blue-400 text-sm mb-4">
              <Zap size={14} />
              Built by Devin — Cognition AI
            </div>
            <h2 className="text-4xl font-bold text-white mb-4 leading-tight">
              Replacing Manual Workflows with{" "}
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-emerald-400 bg-clip-text text-transparent">
                AI-Powered Engineering Tools
              </span>
            </h2>
            <p className="text-lg text-slate-400 leading-relaxed">
              This toolkit demonstrates how Devin accelerates NASA-JPL's most
              common science and engineering workflows. Each module includes
              real use cases pulled from public NASA/NASA-JPL repositories,
              showing the <strong className="text-slate-200">before</strong>{" "}
              (manual process), what{" "}
              <strong className="text-slate-200">Devin built</strong>, and the{" "}
              <strong className="text-slate-200">after</strong> (automated
              pipeline with full code).
            </p>
          </div>

          {/* Module Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            {modules.map((mod) => (
              <button
                key={mod.id}
                onClick={() => setActiveModule(mod.id)}
                className={`p-5 rounded-xl border text-left transition-all duration-200 cursor-pointer ${
                  activeModule === mod.id
                    ? "border-slate-600 bg-slate-800/60 shadow-lg"
                    : "border-slate-700/50 bg-slate-900/50 hover:border-slate-600/50 hover:bg-slate-800/30"
                }`}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{
                      backgroundColor: `${mod.color}15`,
                      color: mod.color,
                    }}
                  >
                    {iconMap[mod.icon]}
                  </div>
                  <div>
                    <h3 className="font-semibold text-white text-sm">
                      {mod.title}
                    </h3>
                    <p className="text-xs text-slate-400">{mod.subtitle}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500">
                    {mod.useCases.length} use cases
                  </span>
                  <span
                    className={`text-xs font-medium px-2 py-0.5 rounded ${
                      activeModule === mod.id
                        ? "bg-blue-900/30 text-blue-400"
                        : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {activeModule === mod.id ? "Active" : "View"}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {modules.map((mod) => (
          <ModuleSection
            key={mod.id}
            module={mod}
            isActive={activeModule === mod.id}
          />
        ))}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950/80">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">
                NASA-JPL Science & Engineering Toolkit
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Built with Devin by Cognition AI — Demonstrating AI-accelerated
                science and engineering workflows
              </p>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/COG-GTM/NASA-JPL-Cognition"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-slate-400 hover:text-white transition-colors flex items-center gap-1"
              >
                <Github size={14} />
                Repository
              </a>
              <span className="text-slate-700">|</span>
              <span className="text-sm text-slate-500">
                Python 3.10+ | React + Vite + Tailwind
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
