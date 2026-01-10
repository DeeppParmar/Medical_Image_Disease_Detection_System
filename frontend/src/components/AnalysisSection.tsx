import { useState } from "react";
import { Sparkles, RotateCcw } from "lucide-react";
import { Button } from "./ui/button";
import ImageUpload from "./ImageUpload";
import AnalysisResults, { AnalysisResult } from "./AnalysisResults";
import { API_ENDPOINTS } from "@/config";

const AnalysisSection = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult[] | null>(null);

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setResults(null);
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResults(null);
    setIsAnalyzing(false);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    
    setIsAnalyzing(true);
    setResults(null);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      
      // Call backend API
      const response = await fetch(API_ENDPOINTS.ANALYZE, { 
        method: 'POST', 
        body: formData 
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Ensure data is an array
      if (Array.isArray(data)) {
        setResults(data);
      } else {
        console.error("Unexpected API response format:", data);
        throw new Error("Invalid response format from API");
      }
    } catch (error) {
      console.error("Analysis failed:", error);
      // Show error to user (you can add toast notification here)
      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setResults(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section className="py-16" id="analyze">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
            Start <span className="gradient-text">Analysis</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload A Chest X-Ray, CT Scan, Or Radiograph For AI-Powered Disease Detection
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Upload Section */}
          <div className="space-y-6">
            <ImageUpload
              onImageSelect={handleImageSelect}
              selectedImage={selectedImage}
              onClear={handleClear}
            />
            
            <div className="flex gap-4">
              <Button
                variant="glow"
                size="lg"
                className="flex-1"
                onClick={handleAnalyze}
                disabled={!selectedImage || isAnalyzing}
              >
                <Sparkles className="w-5 h-5" />
                {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
              </Button>
              
              {(selectedImage || results) && (
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleClear}
                  disabled={isAnalyzing}
                >
                  <RotateCcw className="w-5 h-5" />
                  Reset
                </Button>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div>
            <AnalysisResults results={results} isAnalyzing={isAnalyzing} />
          </div>
        </div>
      </div>
    </section>
  );
};

export default AnalysisSection;
