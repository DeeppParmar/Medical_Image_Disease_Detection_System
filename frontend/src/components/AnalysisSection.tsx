import { useEffect, useState } from "react";
import { Sparkles, RotateCcw, Stethoscope, Bone, FlaskConical, Wand2, Cpu, Clock } from "lucide-react";
import { Button } from "./ui/button";
import ImageUpload from "./ImageUpload";
import AnalysisResults, { AnalysisResult } from "./AnalysisResults";
import { API_ENDPOINTS } from "@/config";

type ScanType = 'auto' | 'chest' | 'bone' | 'tb';

interface ScanTypeOption {
  id: ScanType;
  label: string;
  description: string;
  icon: React.ElementType;
  model: string;
  color: string;
}

const scanTypeOptions: ScanTypeOption[] = [
  {
    id: 'auto',
    label: 'All Detect',
    description: 'Run all 3 models combined',
    icon: Wand2,
    model: 'Combined',
    color: 'from-purple-500 to-pink-500'
  },
  {
    id: 'chest',
    label: 'Chest X-Ray',
    description: '14 disease detection',
    icon: Stethoscope,
    model: 'CheXNet',
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'bone',
    label: 'Bone X-Ray',
    description: 'Musculoskeletal analysis',
    icon: Bone,
    model: 'MURA',
    color: 'from-orange-500 to-amber-500'
  },
  {
    id: 'tb',
    label: 'TB Screening',
    description: 'Tuberculosis detection',
    icon: FlaskConical,
    model: 'TuberculosisNet',
    color: 'from-green-500 to-emerald-500'
  }
];

const AnalysisSection = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult[] | null>(null);
  const [scanType, setScanType] = useState<ScanType>('auto');
  const [modelUsed, setModelUsed] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3>(1);

  const runAnalysis = async (file: File) => {
    setIsAnalyzing(true);
    setResults(null);
    setModelUsed(null);
    setProcessingTime(null);
    setCurrentStep(3);

    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append('image', file);

      // For "All Detect" mode, run all three models
      if (scanType === 'auto') {
        const models = ['chexnet', 'mura', 'tuberculosis'];
        const allResults: AnalysisResult[] = [];
        
        for (const model of models) {
          const modelFormData = new FormData();
          modelFormData.append('image', file);
          
          if (model === 'tuberculosis') {
            modelFormData.append('scan_type', 'chest');
            modelFormData.append('model', 'tuberculosis');
          } else if (model === 'mura') {
            modelFormData.append('scan_type', 'bone');
          } else if (model === 'chexnet') {
            modelFormData.append('scan_type', 'chest');
          }
          
          try {
            const response = await fetch(API_ENDPOINTS.ANALYZE, {
              method: 'POST',
              body: modelFormData
            });
            
            const data = await response.json().catch(() => null);
            if (Array.isArray(data)) {
              allResults.push(...data);
            }
          } catch (modelError) {
            console.error(`Error running ${model}:`, modelError);
          }
        }
        
        const endTime = Date.now();
        setProcessingTime(endTime - startTime);
        setModelUsed('Combined (CheXNet + MURA + TB)');
        setResults(allResults.length > 0 ? allResults : [{
          disease: 'Analysis Complete',
          confidence: 0,
          status: 'info',
          description: 'All models completed but no significant findings detected.'
        }]);
        setIsAnalyzing(false);
        return;
      }

      if (scanType === 'tb') {
        formData.append('scan_type', 'chest');
        formData.append('model', 'tuberculosis');
      } else if (scanType !== 'auto') {
        formData.append('scan_type', scanType);
      }

      const name = file.name.toLowerCase();
      const isDicom = name.endsWith('.dcm');
      const boneKeywords = ['wrist', 'hand', 'elbow', 'shoulder', 'humerus', 'finger', 'forearm', 'ankle', 'foot', 'knee', 'hip', 'bone', 'mura'];
      const isBone = boneKeywords.some((k) => name.includes(k));
      if (isDicom) formData.append('model', 'rsna');
      else if (isBone && scanType === 'auto') formData.append('model', 'mura');

      const response = await fetch(API_ENDPOINTS.ANALYZE, {
        method: 'POST',
        body: formData
      });

      const endTime = Date.now();
      setProcessingTime(endTime - startTime);

      const usedModel = response.headers.get('X-Model-Used');
      if (usedModel) {
        const modelNames: Record<string, string> = {
          'chexnet': 'CheXNet',
          'mura': 'MURA',
          'tuberculosis': 'TuberculosisNet',
          'rsna': 'RSNA',
          'unet': 'UNet'
        };
        setModelUsed(modelNames[usedModel] || usedModel);
      }

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        if (Array.isArray(data)) {
          setResults(data);
          return;
        }
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      if (Array.isArray(data)) {
        setResults(data);
      } else {
        console.error('Unexpected API response format:', data);
        throw new Error('Invalid response format from API');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setResults([{
        disease: 'Connection Error',
        confidence: 0,
        status: 'warning',
        description: `Could not connect to backend. Make sure the server is running at ${API_ENDPOINTS.ANALYZE}`
      }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setCurrentStep(2);
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResults(null);
    setIsAnalyzing(false);
    setModelUsed(null);
    setProcessingTime(null);
    setCurrentStep(1);
    setScanType('auto');
  };

  return (
    <section className="py-8 md:py-12 lg:py-16 bg-background/50" id="analyze">
      <div className="container mx-auto px-3 md:px-6">
        <div className="text-center mb-6 md:mb-10">
          <h2 className="font-display text-3xl md:text-4xl lg:text-5xl font-bold mb-3 md:mb-4 tracking-tight px-2">
            AI-Powered <span className="gradient-text">Disease Detection</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-xs md:text-sm lg:text-base px-4 leading-relaxed">
            Upload medical image → Select AI model → Get instant diagnosis
          </p>
        </div>

        {/* Steps Progress Indicator with Labels */}
        <div className="mb-8 md:mb-10 px-2">
          <div className="flex items-center justify-center max-w-2xl mx-auto">
            {[
              { num: 1, label: 'Upload', sublabel: 'Image' },
              { num: 2, label: 'Select', sublabel: 'Model' },
              { num: 3, label: 'Analyze', sublabel: 'Results' }
            ].map((step, idx) => (
              <div key={step.num} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div className={`
                    w-10 h-10 md:w-12 md:h-12 rounded-full flex items-center justify-center font-bold transition-all duration-300 mb-2
                    ${currentStep === step.num ? 'bg-primary text-primary-foreground scale-110 shadow-lg shadow-primary/30 ring-4 ring-primary/20' :
                      currentStep > step.num ? 'bg-primary/30 text-primary border-2 border-primary/40' : 'bg-secondary/60 text-muted-foreground border-2 border-border/60'}
                  `}>
                    {step.num}
                  </div>
                  <div className="text-center">
                    <p className={`text-xs md:text-sm font-semibold transition-colors ${currentStep === step.num ? 'text-primary' : 'text-muted-foreground'}`}>
                      {step.label}
                    </p>
                    <p className="text-[10px] md:text-xs text-muted-foreground/60">{step.sublabel}</p>
                  </div>
                </div>
                {idx < 2 && (
                  <div className={`h-0.5 md:h-1 flex-1 mx-1 md:mx-3 rounded-full transition-all duration-300 max-w-[60px] md:max-w-[100px] ${currentStep > step.num ? 'bg-primary/40' : 'bg-border/50'}`} />
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="max-w-5xl mx-auto px-2 md:px-4">
          {/* Step 1: Upload Image */}
          {currentStep === 1 && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-card/50 border border-border/50 rounded-2xl md:rounded-3xl p-4 md:p-6 lg:p-8 backdrop-blur-sm shadow-xl">
                <div className="mb-4">
                  <h3 className="font-display text-lg md:text-xl font-bold mb-1 flex items-center gap-2">
                    <span className="inline-block w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center text-sm font-bold">1</span>
                    Upload Medical Image
                  </h3>
                  <p className="text-xs md:text-sm text-muted-foreground ml-10">Drop your X-ray or medical scan to begin</p>
                </div>
                <ImageUpload
                  onImageSelect={handleImageSelect}
                  selectedImage={selectedImage}
                  onClear={handleClear}
                />
              </div>
            </div>
          )}

          {/* Step 2: Select Model */}
          {currentStep === 2 && selectedImage && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6 md:space-y-8">
              {/* Image Preview Card */}
              <div className="bg-card/50 border border-border/50 rounded-2xl md:rounded-3xl p-4 md:p-6 backdrop-blur-sm shadow-xl">
                <div className="mb-4">
                  <h3 className="font-display text-lg md:text-xl font-bold mb-1 flex items-center gap-2">
                    <span className="inline-block w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center text-sm font-bold">✓</span>
                    Image Uploaded
                  </h3>
                  <p className="text-xs md:text-sm text-muted-foreground ml-10">{selectedImage.name} · {(selectedImage.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <div className="relative rounded-xl md:rounded-2xl overflow-hidden border border-border/50 max-w-sm mx-auto">
                  <div className="aspect-square bg-black/90 relative">
                    <img
                      src={URL.createObjectURL(selectedImage)}
                      alt="Preview"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-0 left-4 right-4 h-0.5 bg-gradient-to-r from-transparent via-primary/60 to-transparent animate-scan" />
                  </div>
                </div>
              </div>

              {/* Model Selection */}
              <div className="bg-card/50 border border-border/50 rounded-2xl md:rounded-3xl p-4 md:p-6 backdrop-blur-sm shadow-xl">
                <div className="mb-4 md:mb-6">
                  <h3 className="font-display text-lg md:text-xl font-bold mb-1 flex items-center gap-2">
                    <span className="inline-block w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center text-sm font-bold">2</span>
                    Select AI Model
                  </h3>
                  <p className="text-xs md:text-sm text-muted-foreground ml-10">Choose the analysis type for your medical image</p>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
                  {scanTypeOptions.map((option) => {
                    const Icon = option.icon;
                    const isSelected = scanType === option.id;
                    return (
                      <button
                        key={option.id}
                        onClick={() => setScanType(option.id)}
                        className={`relative p-4 md:p-6 rounded-xl md:rounded-2xl border transition-all duration-300 text-left group hover:scale-[1.02]
                          ${isSelected
                            ? 'border-primary bg-primary/10 ring-2 ring-primary/20 shadow-lg shadow-primary/10'
                            : 'border-border/50 bg-card/30 hover:border-primary/30 hover:bg-card/50'
                          }
                        `}
                      >
                        <div className={`w-12 h-12 md:w-14 md:h-14 rounded-xl md:rounded-2xl bg-secondary flex items-center justify-center mb-3 md:mb-4 transition-all duration-300 border border-border/50
                          ${isSelected ? 'bg-primary/20 border-primary/20 scale-105' : 'group-hover:scale-105'}`}>
                          <Icon className={`w-5 h-5 md:w-6 md:h-6 ${isSelected ? 'text-primary' : 'text-muted-foreground'}`} />
                        </div>
                        <h3 className="font-bold text-sm md:text-base mb-1 md:mb-2">{option.label}</h3>
                        <p className="text-[10px] md:text-xs text-muted-foreground leading-relaxed">{option.description}</p>
                        {isSelected && (
                          <div className="absolute top-3 md:top-4 right-3 md:right-4 flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                            <div className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse delay-75" />
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 md:gap-4 justify-center items-stretch sm:items-center px-2">
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => {
                    setCurrentStep(1);
                    setResults(null);
                    setModelUsed(null);
                    setProcessingTime(null);
                  }}
                  className="w-full sm:w-auto min-w-[140px] rounded-xl border-border/50 hover:border-primary/30"
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Change Image
                </Button>
                <Button
                  variant="glow"
                  size="lg"
                  onClick={() => selectedImage && runAnalysis(selectedImage)}
                  disabled={!scanType}
                  className="w-full sm:w-auto min-w-[200px] rounded-xl shadow-lg shadow-primary/20 hover:shadow-primary/30"
                >
                  <Sparkles className="w-5 h-5 mr-2" />
                  Analyze with {scanTypeOptions.find(o => o.id === scanType)?.model}
                </Button>
              </div>
            </div>
          )}

          {/* Step 3: Results */}
          {currentStep === 3 && (
            <div className="animate-in fade-in zoom-in-95 duration-500">
              <div className="grid lg:grid-cols-12 gap-4 md:gap-6 lg:gap-8 items-start">
                {/* Image Preview Sidebar */}
                <div className="lg:col-span-5 space-y-3 md:space-y-4">
                  <div className="glass-card rounded-xl md:rounded-2xl overflow-hidden border border-border/50 shadow-xl">
                    <div className="aspect-square bg-black relative">
                      {selectedImage && (
                        <img
                          src={URL.createObjectURL(selectedImage)}
                          alt="Analyzed image"
                          className="w-full h-full object-contain"
                        />
                      )}
                      <div className="absolute top-0 left-4 right-4 h-0.5 bg-gradient-to-r from-transparent via-primary/60 to-transparent animate-scan" />
                    </div>
                    <div className="p-3 md:p-4 bg-card/80 border-t border-border/50">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <Cpu className="w-4 h-4 text-primary shrink-0" />
                          <span className="text-xs md:text-sm font-medium truncate">Model: {modelUsed || 'Initializing...'}</span>
                        </div>
                        {processingTime && (
                          <div className="flex items-center gap-2">
                            <Clock className="w-4 h-4 text-muted-foreground shrink-0" />
                            <span className="text-xs md:text-sm text-muted-foreground">{(processingTime / 1000).toFixed(2)}s</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <Button
                    variant="outline"
                    className="w-full rounded-xl border-border/50"
                    onClick={handleClear}
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    New Analysis
                  </Button>
                </div>

                {/* Results Section */}
                <div className="lg:col-span-7">
                  <AnalysisResults results={results} isAnalyzing={isAnalyzing} modelUsed={modelUsed} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default AnalysisSection;

