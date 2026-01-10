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
    if (files.length > 0) {
      const f = files[0];
      const isDicom = f.name.toLowerCase().endsWith('.dcm');
      const isImage = f.type.startsWith('image/');
      if (isImage || isDicom) {
        handleFileSelect(f);
      }
    }
  }, []);

  const handleFileSelect = (file: File) => {
    onImageSelect(file);
    const isDicom = file.name.toLowerCase().endsWith('.dcm');
    if (isDicom) {
      setPreview(null);
      return;
    }
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
          className={`relative border-2 border-dashed rounded-2xl md:rounded-3xl p-6 md:p-8 lg:p-12 text-center transition-all duration-300 cursor-pointer group
            ${isDragOver
              ? 'border-primary bg-primary/5 scale-[1.01]'
              : 'border-border/60 hover:border-primary/50 hover:bg-card/50'
            }`}
        >
          <input
            type="file"
            accept="image/*,.dcm"
            onChange={handleInputChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />

          <div className="flex flex-col items-center gap-3 md:gap-4">
            <div className={`p-3 md:p-4 lg:p-5 rounded-2xl transition-all duration-300 ${isDragOver ? 'bg-primary/20' : 'bg-secondary group-hover:bg-primary/10'}`}>
              <Upload className={`w-6 h-6 md:w-8 md:h-8 lg:w-10 lg:h-10 transition-colors ${isDragOver ? 'text-primary' : 'text-muted-foreground group-hover:text-primary'}`} />
            </div>

            <div className="space-y-1.5 md:space-y-2">
              <p className="font-display font-semibold text-sm md:text-base lg:text-xl mb-1 px-2">
                {isDragOver ? 'Drop Your Image Here' : 'Upload Medical Image'}
              </p>
              <p className="text-xs md:text-sm text-muted-foreground">
                Drag & Drop Or Click To Browse
              </p>
              <div className="bg-secondary/40 rounded-lg p-2 md:p-3 mt-3 md:mt-4 mx-2">
                <p className="text-[10px] md:text-xs text-muted-foreground uppercase tracking-widest font-medium">
                  Supported Formats
                </p>
                <p className="text-[10px] md:text-xs text-primary font-medium mt-1">
                  X-Ray • CT Scan • Radiograph (PNG, JPG, DICOM)
                </p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center gap-2 md:gap-4 mt-2">
              <div className="flex items-center gap-2 text-[10px] md:text-xs text-muted-foreground">
                <FileImage className="w-3 h-3 md:w-4 md:h-4" />
                <span>Max File Size: 10MB</span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative rounded-xl md:rounded-2xl overflow-hidden glass-card border border-border/50 shadow-lg">
          <div className="aspect-square max-h-[300px] sm:max-h-[350px] md:max-h-[450px] relative bg-black/90">
            {preview ? (
              <img
                src={preview}
                alt="Uploaded medical image"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-sm text-muted-foreground">
                <div className="flex flex-col items-center gap-3">
                  <FileImage className="w-6 h-6 md:w-8 md:h-8 opacity-20" />
                  <span className="text-xs md:text-sm">Preview Not Available</span>
                </div>
              </div>
            )}

            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 border-2 border-primary/20 rounded-lg m-3 md:m-4" />
              <div className="absolute top-0 left-3 right-3 md:left-4 md:right-4 h-0.5 md:h-1 bg-gradient-to-r from-transparent via-primary/60 to-transparent animate-scan" />
            </div>
          </div>

          <div className="p-3 md:p-4 flex items-center justify-between border-t border-border/50 bg-card/60">
            <div className="flex items-center gap-2 md:gap-3 min-w-0 flex-1">
              <div className="w-7 h-7 md:w-8 md:h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <Image className="w-3.5 h-3.5 md:w-4 md:h-4 text-primary" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-xs md:text-sm truncate">{selectedImage.name}</p>
                <p className="text-[10px] md:text-xs text-muted-foreground">
                  {(selectedImage.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>

            <Button variant="outline" size="icon" onClick={handleClear} className="h-7 w-7 md:h-8 md:w-8 rounded-full border-border/50 shrink-0">
              <X className="w-3 h-3 md:w-3.5 md:h-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
