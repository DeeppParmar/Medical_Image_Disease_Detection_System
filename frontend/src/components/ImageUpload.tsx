import { useState, useCallback } from "react";
import { Upload, Image, X, FileImage } from "lucide-react";
import { Button } from "./ui/button";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  onClear: () => void;
}

const ImageUpload = ({ onImageSelect, selectedImage, onClear }: ImageUploadProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleFileSelect(files[0]);
    }
  }, []);

  const handleFileSelect = (file: File) => {
    onImageSelect(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleClear = () => {
    setPreview(null);
    onClear();
  };

  return (
    <div className="w-full">
      {!selectedImage ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer group
            ${isDragOver 
              ? 'border-primary bg-primary/5 scale-[1.02]' 
              : 'border-border hover:border-primary/50 hover:bg-card/50'
            }`}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleInputChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <div className="flex flex-col items-center gap-4">
            <div className={`p-4 rounded-2xl transition-all duration-300 ${
              isDragOver ? 'bg-primary/20' : 'bg-secondary group-hover:bg-primary/10'
            }`}>
              <Upload className={`w-10 h-10 transition-colors ${
                isDragOver ? 'text-primary' : 'text-muted-foreground group-hover:text-primary'
              }`} />
            </div>
            
            <div>
              <p className="font-display font-semibold text-lg mb-1">
                {isDragOver ? 'Drop Your Image Here' : 'Upload Medical Image'}
              </p>
              <p className="text-sm text-muted-foreground">
                Drag & Drop Or Click To Browse
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Supports: X-Ray, CT Scan, Radiograph (PNG, JPG, DICOM)
              </p>
            </div>

            <div className="flex items-center gap-4 mt-2">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <FileImage className="w-4 h-4" />
                <span>Max 10MB</span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative rounded-2xl overflow-hidden glass-card">
          <div className="aspect-square max-h-[400px] relative bg-black/50">
            <img
              src={preview || ''}
              alt="Uploaded medical image"
              className="w-full h-full object-contain"
            />
            
            {/* Scan Effect Overlay */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 border-2 border-primary/30 rounded-lg m-4" />
              <div className="absolute top-0 left-4 right-4 h-1 bg-gradient-to-r from-transparent via-primary/50 to-transparent animate-scan" />
            </div>
          </div>
          
          <div className="p-4 flex items-center justify-between border-t border-border/50">
            <div className="flex items-center gap-3">
              <Image className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium text-sm truncate max-w-[200px]">{selectedImage.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(selectedImage.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            
            <Button variant="ghost" size="icon" onClick={handleClear}>
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
