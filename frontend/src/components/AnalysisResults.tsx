import { AlertTriangle, CheckCircle, Activity, Target, Brain, Info, Cpu } from "lucide-react";
import { Progress } from "./ui/progress";

export interface AnalysisResult {
  disease: string;
  confidence: number;
  status: 'healthy' | 'warning' | 'critical';
  description: string;
  regions?: string[];
}

interface AnalysisResultsProps {
  results: AnalysisResult[] | null;
  isAnalyzing: boolean;
  modelUsed?: string | null;
}

const statusConfig = {
  healthy: {
    icon: CheckCircle,
    color: 'text-success',
    bgColor: 'bg-success/10',
    borderColor: 'border-success/30',
    progressColor: 'bg-success',
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/30',
    progressColor: 'bg-warning',
  },
  critical: {
    icon: AlertTriangle,
    color: 'text-destructive',
    bgColor: 'bg-destructive/10',
    borderColor: 'border-destructive/30',
    progressColor: 'bg-destructive',
  },
};

const AnalysisResults = ({ results, isAnalyzing, modelUsed }: AnalysisResultsProps) => {
  if (isAnalyzing) {
    return (
      <div className="glass-card rounded-2xl p-8">
        <div className="flex flex-col items-center justify-center py-12 gap-6">
          <div className="relative">
            <div className="w-20 h-20 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
            <Brain className="absolute inset-0 m-auto w-8 h-8 text-primary animate-pulse" />
          </div>
          <div className="text-center">
            <p className="font-display font-semibold text-lg mb-2">Analyzing Image...</p>
            <p className="text-sm text-muted-foreground">AI Model Processing Medical Data</p>
          </div>
          <div className="w-full max-w-xs">
            <div className="h-2 rounded-full bg-secondary overflow-hidden">
              <div className="h-full bg-primary animate-shimmer" style={{ width: '60%' }} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="glass-card rounded-2xl p-8">
        <div className="flex flex-col items-center justify-center py-12 gap-4 text-center">
          <div className="p-4 rounded-2xl bg-secondary">
            <Activity className="w-10 h-10 text-muted-foreground" />
          </div>
          <div>
            <p className="font-display font-semibold text-lg mb-2">No Analysis Yet</p>
            <p className="text-sm text-muted-foreground">Upload A Medical Image To Begin AI Analysis</p>
          </div>
        </div>
      </div>
    );
  }

  const primaryResult = results[0];
  const config = statusConfig[primaryResult.status];
  const StatusIcon = config.icon;

  return (
    <div className="space-y-4">
      {/* Primary Result */}
      <div className={`glass-card rounded-2xl p-6 border ${config.borderColor}`}>
        <div className="flex items-start gap-4 mb-6">
          <div className={`p-3 rounded-xl ${config.bgColor}`}>
            <StatusIcon className={`w-6 h-6 ${config.color}`} />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1">
              <h3 className="font-display font-bold text-xl">{primaryResult.disease}</h3>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${config.bgColor} ${config.color}`}>
                {primaryResult.status === 'healthy' ? 'Normal' : primaryResult.status === 'warning' ? 'Detected' : 'High Risk'}
              </span>
            </div>
            <p className="text-sm text-muted-foreground">{primaryResult.description}</p>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Confidence Score</span>
            <span className="font-semibold">{primaryResult.confidence}%</span>
          </div>
          <Progress value={primaryResult.confidence} className="h-3" />
        </div>

        {primaryResult.regions && primaryResult.regions.length > 0 && (
          <div className="mt-4 pt-4 border-t border-border/50">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">Affected Regions</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {primaryResult.regions.map((region, index) => (
                <span key={index} className="px-3 py-1 rounded-full bg-secondary text-xs font-medium">
                  {region}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Secondary Results */}
      {results.slice(1).map((result, index) => {
        const resultConfig = statusConfig[result.status];
        const ResultIcon = resultConfig.icon;

        return (
          <div key={index} className="glass-card rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${resultConfig.bgColor}`}>
                <ResultIcon className={`w-4 h-4 ${resultConfig.color}`} />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{result.disease}</span>
                  <span className="text-sm text-muted-foreground">{result.confidence}%</span>
                </div>
                <Progress value={result.confidence} className="h-1.5 mt-2" />
              </div>
            </div>
          </div>
        );
      })}

      {/* Disclaimer */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-primary/5 border border-primary/20">
        <Info className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
        <p className="text-xs text-muted-foreground">
          This AI Analysis Is For Research And Educational Purposes Only. Always Consult A Qualified Healthcare Professional For Medical Diagnosis And Treatment.
        </p>
      </div>
    </div>
  );
};

export default AnalysisResults;
