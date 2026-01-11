import { AlertTriangle, CheckCircle, Activity, Target, Brain, Info, Cpu, Bone, Heart, Stethoscope } from "lucide-react";
import { Progress } from "./ui/progress";

export interface AnalysisResult {
  disease: string;
  confidence: number;
  status: 'healthy' | 'warning' | 'critical';
  description: string;
  regions?: string[];
  enhanced_detection?: boolean;
}

interface AnalysisResultsProps {
  results: AnalysisResult[] | null;
  isAnalyzing: boolean;
  modelUsed?: string | null;
}

// Model-specific configurations
const modelConfig: Record<string, { icon: any; color: string; title: string; subtitle: string }> = {
  'CheXNet': {
    icon: Heart,
    color: 'text-red-500',
    title: 'Chest X-Ray Analysis',
    subtitle: '14-disease detection model'
  },
  'TuberculosisNet': {
    icon: Stethoscope,
    color: 'text-orange-500',
    title: 'Tuberculosis Screening',
    subtitle: 'TB-specific detection model'
  },
  'MURA': {
    icon: Bone,
    color: 'text-blue-500',
    title: 'Bone X-Ray Analysis',
    subtitle: 'Musculoskeletal abnormality detection'
  },
  'RSNA': {
    icon: Brain,
    color: 'text-purple-500',
    title: 'Brain CT Analysis',
    subtitle: 'Intracranial hemorrhage detection'
  }
};

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
  // Get model-specific config
  const currentModelConfig = modelUsed ? modelConfig[modelUsed] : null;
  const ModelIcon = currentModelConfig?.icon || Brain;

  if (isAnalyzing) {
    return (
      <div className="glass-card rounded-xl md:rounded-2xl p-6 md:p-8 shadow-xl">
        <div className="flex flex-col items-center justify-center py-8 md:py-12 gap-4 md:gap-6">
          <div className="relative">
            <div className="w-16 h-16 md:w-20 md:h-20 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
            <Brain className="absolute inset-0 m-auto w-6 h-6 md:w-8 md:h-8 text-primary animate-pulse" />
          </div>
          <div className="text-center px-4">
            <p className="font-display font-semibold text-base md:text-lg mb-2">Analyzing Image...</p>
            <p className="text-xs md:text-sm text-muted-foreground">AI Model Processing Medical Data</p>
          </div>
          <div className="w-full max-w-xs px-4">
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
      <div className="glass-card rounded-xl md:rounded-2xl p-6 md:p-8 shadow-xl">
        <div className="flex flex-col items-center justify-center py-8 md:py-12 gap-4 text-center px-4">
          <div className="p-3 md:p-4 rounded-2xl bg-secondary">
            <Activity className="w-8 h-8 md:w-10 md:h-10 text-muted-foreground" />
          </div>
          <div>
            <p className="font-display font-semibold text-base md:text-lg mb-2">Ready to Analyze</p>
            <p className="text-xs md:text-sm text-muted-foreground">Select a model to start AI analysis</p>
          </div>
        </div>
      </div>
    );
  }

  const primaryResult = results[0];
  const config = statusConfig[primaryResult.status];
  const StatusIcon = config.icon;

  return (
    <div className="space-y-3 md:space-y-4">
      {/* Model Header */}
      {currentModelConfig && (
        <div className="flex items-center gap-3 px-2 mb-2">
          <div className={`p-2 rounded-lg bg-primary/10`}>
            <ModelIcon className={`w-5 h-5 ${currentModelConfig.color}`} />
          </div>
          <div>
            <h3 className="font-semibold text-sm">{currentModelConfig.title}</h3>
            <p className="text-xs text-muted-foreground">{currentModelConfig.subtitle}</p>
          </div>
        </div>
      )}
      
      <div className={`glass-card rounded-xl md:rounded-2xl p-4 md:p-6 border ${config.borderColor} shadow-xl`}>
        <div className="flex items-start gap-3 md:gap-4 mb-4 md:mb-6">
          <div className={`p-2.5 md:p-3 rounded-xl ${config.bgColor} shrink-0`}>
            <StatusIcon className={`w-5 h-5 md:w-6 md:h-6 ${config.color}`} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1 md:mb-2">
              <h3 className="font-display font-bold text-lg md:text-xl">{primaryResult.disease}</h3>
              <span className={`px-2 py-0.5 rounded-full text-[10px] md:text-xs font-medium ${config.bgColor} ${config.color} whitespace-nowrap`}>
                {primaryResult.status === 'healthy' ? 'Normal' : primaryResult.status === 'warning' ? 'Detected' : 'High Risk'}
              </span>
            </div>
            <p className="text-xs md:text-sm text-muted-foreground leading-relaxed">{primaryResult.description}</p>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs md:text-sm">
            <span className="text-muted-foreground">Confidence Score</span>
            <span className="font-semibold">{primaryResult.confidence}%</span>
          </div>
          <Progress value={primaryResult.confidence} className="h-2 md:h-3" />
        </div>

        {primaryResult.regions && primaryResult.regions.length > 0 && (
          <div className="mt-4 pt-4 border-t border-border/50">
            <div className="flex items-center gap-2 mb-2 md:mb-3">
              <Target className="w-3.5 h-3.5 md:w-4 md:h-4 text-primary" />
              <span className="text-xs md:text-sm font-medium">Affected Regions</span>
            </div>
            <div className="flex flex-wrap gap-1.5 md:gap-2">
              {primaryResult.regions.map((region, index) => (
                <span key={index} className="px-2 md:px-3 py-1 rounded-full bg-secondary text-[10px] md:text-xs font-medium">
                  {region}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {results.slice(1).map((result, index) => {
        const resultConfig = statusConfig[result.status];
        const ResultIcon = resultConfig.icon;

        return (
          <div key={index} className="glass-card rounded-lg md:rounded-xl p-3 md:p-4 shadow-lg">
            <div className="flex items-center gap-2 md:gap-3">
              <div className={`p-1.5 md:p-2 rounded-lg ${resultConfig.bgColor} shrink-0`}>
                <ResultIcon className={`w-3.5 h-3.5 md:w-4 md:h-4 ${resultConfig.color}`} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2 mb-1.5 md:mb-2">
                  <span className="font-medium text-xs md:text-sm truncate">{result.disease}</span>
                  <span className="text-xs md:text-sm text-muted-foreground shrink-0">{result.confidence}%</span>
                </div>
                <Progress value={result.confidence} className="h-1.5 md:h-2" />
              </div>
            </div>
          </div>
        );
      })}

      <div className="flex items-start gap-2 md:gap-3 p-3 md:p-4 rounded-lg md:rounded-xl bg-primary/5 border border-primary/20">
        <Info className="w-4 h-4 md:w-5 md:h-5 text-primary flex-shrink-0 mt-0.5" />
        <p className="text-[10px] md:text-xs text-muted-foreground leading-relaxed">
          This AI Analysis Is For Research And Educational Purposes Only. Always Consult A Qualified Healthcare Professional For Medical Diagnosis And Treatment.
        </p>
      </div>
    </div>
  );
};

export default AnalysisResults;
