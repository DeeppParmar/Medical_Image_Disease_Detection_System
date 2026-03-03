import { useState } from "react";
import { RotateCcw, Cpu, Clock } from "lucide-react";
import { Button } from "./ui/button";
import ImageUpload from "./ImageUpload";
import AnalysisResults, { AnalysisResult, ErrorState } from "./AnalysisResults";
import { API_ENDPOINTS } from "@/config";

const AnalysisSection = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult[] | null>(null);
  const [modelUsed, setModelUsed] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [errorState, setErrorState] = useState<ErrorState | null>(null);

  const runAnalysis = async (file: File) => {
    setIsAnalyzing(true);
    setResults(null);
    setModelUsed(null);
    setProcessingTime(null);
    setErrorState(null);
    setShowResults(true);

    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append('image', file);

      // Auto-detect model based on filename
      const name = file.name.toLowerCase();
      const isDicom = name.endsWith('.dcm');
      const boneKeywords = ['wrist', 'hand', 'elbow', 'shoulder', 'humerus', 'finger', 'forearm', 'ankle', 'foot', 'knee', 'hip', 'bone', 'mura', 'fracture'];
      const isBone = boneKeywords.some((k) => name.includes(k));
      const tbKeywords = ['tb', 'tuberculosis'];
      const isTB = tbKeywords.some((k) => name.includes(k));

      // Set appropriate scan_type and model based on filename hints
      if (isDicom) {
        formData.append('model', 'rsna');
        formData.append('scan_type', 'ct');
      } else if (isBone) {
        formData.append('scan_type', 'bone');
        formData.append('model', 'mura');
      } else if (isTB) {
        formData.append('scan_type', 'chest');
        formData.append('model', 'tuberculosis');
      } else {
        // Default: let backend auto-detect (will try TB then CheXNet for chest X-rays)
        formData.append('scan_type', 'auto');
      }

      const response = await fetch(API_ENDPOINTS.ANALYZE, {
        method: 'POST',
        body: formData
      });

      const endTime = Date.now();
      setProcessingTime(endTime - startTime);

      const usedModel = response.headers.get('X-Model-Used');
      console.log('🔬 Analysis Response:');
      console.log('   Model Used:', usedModel);

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
      console.log('   Response Status:', response.status);
      console.log('   Results:', JSON.stringify(data, null, 2));

      // ── Feature 2 — Structured error handling ────────────────────
      if (!response.ok) {
        const errorField = data?.error as string | undefined;

        if (response.status === 503 || errorField === 'model_not_loaded') {
          setErrorState({
            type: 'model_unavailable',
            message: data?.message || 'Model not available. The AI model for this scan type is not loaded. Please try again or select a different scan.',
          });
          return;
        }
        // Medical image validation rejection
        if (response.status === 422 && (errorField === 'not_medical_image' || data?.reason === 'non_medical_image')) {
          setErrorState({
            type: 'not_medical_image',
            message: data?.message || 'Uploaded image does not appear to be a valid medical scan. Please upload an X-ray, CT scan, or MRI.',
            validationDetails: {
              heuristicScore: data?.heuristic_score ?? null,
              cnnProbability: data?.cnn_probability ?? null,
              subReason: data?.sub_reason ?? null,
            },
          });
          return;
        }
        if (response.status === 415 || response.status === 422 || errorField === 'unsupported_scan' || errorField === 'resolution_too_low') {
          setErrorState({
            type: 'unsupported_scan',
            message: data?.message || 'Unsupported scan type. This image format or scan type cannot be processed.',
          });
          return;
        }
        if (response.status === 413 || errorField === 'file_too_large') {
          setErrorState({
            type: 'unsupported_scan',
            message: data?.message || 'File too large. Maximum server limit is 15MB.',
          });
          return;
        }

        // If it returned an array (legacy unavailable model result), still display it
        if (Array.isArray(data)) {
          setResults(data);
          return;
        }

        // Catch-all
        setErrorState({
          type: 'processing_failed',
          message: data?.message || `Processing failed. An unexpected error occurred during analysis. (HTTP ${response.status})`,
        });
        return;
      }

      if (Array.isArray(data)) {
        // Log each result for verification
        console.log('📊 Analysis Results Summary:');
        data.forEach((r: AnalysisResult, i: number) => {
          console.log(`   ${i + 1}. ${r.disease}: ${r.confidence}% (${r.status})`);
        });
        setResults(data);
      } else {
        console.error('Unexpected API response format:', data);
        setErrorState({
          type: 'processing_failed',
          message: 'Invalid response format from API.',
        });
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setErrorState({
        type: 'processing_failed',
        message: `Could not connect to backend. Make sure the server is running at ${API_ENDPOINTS.ANALYZE}`,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setErrorState(null);
    // Auto-analyze immediately after image selection
    runAnalysis(file);
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResults(null);
    setIsAnalyzing(false);
    setModelUsed(null);
    setProcessingTime(null);
    setShowResults(false);
    setErrorState(null);
  };

  // Create uploaded image URL for heatmap toggle
  const uploadedImageUrl = selectedImage ? URL.createObjectURL(selectedImage) : null;

  return (
    <section className="py-8 md:py-12 lg:py-16 flex-1 flex flex-col" id="analyze">
      <div className="container mx-auto px-3 md:px-6 flex-1 flex flex-col">
        <div className="text-center mb-6 md:mb-10">
          <div className="inline-flex items-center gap-2 px-3 md:px-4 py-1.5 md:py-2 rounded-full bg-primary/10 border border-primary/20 mb-4 md:mb-6">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-xs md:text-sm font-medium text-primary">AI-Powered Analysis</span>
          </div>
          <h2 className="font-display text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold mb-3 md:mb-4 tracking-tight px-2">
            Medical Image <span className="text-primary">Analysis</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-xs md:text-sm lg:text-base px-4 leading-relaxed">
            Upload your medical image and get instant AI-powered diagnosis
          </p>
        </div>

        <div className="max-w-5xl mx-auto px-2 md:px-4 w-full flex-1">
          {!showResults ? (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-card/50 border border-border/50 rounded-2xl md:rounded-3xl p-4 md:p-6 lg:p-8 backdrop-blur-sm shadow-xl">
                <ImageUpload
                  onImageSelect={handleImageSelect}
                  selectedImage={selectedImage}
                  onClear={handleClear}
                />
              </div>
            </div>
          ) : (
            <div className="animate-in fade-in zoom-in-95 duration-500">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 md:gap-6 lg:gap-8 items-start">
                {/* Image Preview Sidebar */}
                <div className="lg:col-span-5 space-y-3 md:space-y-4">
                  <div className="glass-card rounded-xl md:rounded-2xl overflow-hidden border border-border/50 shadow-xl">
                    <div className="aspect-square max-h-[250px] sm:max-h-[300px] md:max-h-[350px] lg:max-h-none bg-black relative">
                      {selectedImage && (
                        <img
                          src={URL.createObjectURL(selectedImage)}
                          alt="Analyzed image"
                          className="w-full h-full object-contain"
                        />
                      )}
                      {isAnalyzing && (
                        <div className="absolute top-0 left-4 right-4 h-0.5 bg-gradient-to-r from-transparent via-primary/60 to-transparent animate-scan" />
                      )}
                    </div>
                    <div className="p-3 md:p-4 bg-card/80 border-t border-border/50">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex items-center gap-2 min-w-0">
                          <Cpu className="w-4 h-4 text-primary shrink-0" />
                          <span className="text-xs md:text-sm font-medium truncate">
                            {isAnalyzing ? 'Analyzing...' : `Model: ${modelUsed || 'Auto-Detect'}`}
                          </span>
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
                  <AnalysisResults
                    results={results}
                    isAnalyzing={isAnalyzing}
                    modelUsed={modelUsed}
                    errorState={errorState}
                    onReset={handleClear}
                    uploadedImageUrl={uploadedImageUrl}
                  />
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
