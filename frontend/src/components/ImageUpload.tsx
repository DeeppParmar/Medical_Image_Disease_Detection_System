import { useState, useCallback } from "react";
import { Upload, Image, X, FileImage, AlertTriangle, ShieldAlert, CheckCircle } from "lucide-react";
import { Button } from "./ui/button";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  onClear: () => void;
}

// ── Feature 5 — Validation types ────────────────────────────────────
interface ValidationIssue {
  type: 'error' | 'warning';
  message: string;
}

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const MIN_RESOLUTION = 64;

/**
 * Compute Laplacian-like variance on grayscale pixel data from canvas.
 * Uses a 3×3 kernel [0, -1, 0, -1, 4, -1, 0, -1, 0].
 */
function computeLaplacianVariance(imageData: ImageData): number {
  const { width, height, data } = imageData;
  // Convert to grayscale (average RGB)
  const gray = new Float32Array(width * height);
  for (let i = 0; i < gray.length; i++) {
    const idx = i * 4;
    gray[i] = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
  }

  const kernel = [0, -1, 0, -1, 4, -1, 0, -1, 0];
  let sum = 0;
  let sumSq = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let val = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          val += gray[(y + ky) * width + (x + kx)] * kernel[(ky + 1) * 3 + (kx + 1)];
        }
      }
      sum += val;
      sumSq += val * val;
      count++;
    }
  }

  if (count === 0) return 999;
  const mean = sum / count;
  return sumSq / count - mean * mean; // variance
}

const ImageUpload = ({ onImageSelect, selectedImage, onClear }: ImageUploadProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([]);
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  /**
   * Validate a file before allowing upload. Returns true if file should proceed.
   */
  const validateFile = useCallback((file: File): Promise<{ issues: ValidationIssue[]; blocking: boolean }> => {
    return new Promise((resolve) => {
      const issues: ValidationIssue[] = [];
      let blocking = false;

      // 1. File size check
      if (file.size > MAX_FILE_SIZE) {
        issues.push({ type: 'error', message: `File too large. Maximum size is 10MB. (${(file.size / 1024 / 1024).toFixed(1)}MB)` });
        blocking = true;
      }

      // 2. File type check
      const isDicom = file.name.toLowerCase().endsWith('.dcm');
      const isAllowedType = ALLOWED_TYPES.includes(file.type);
      if (!isAllowedType && !isDicom) {
        issues.push({ type: 'error', message: 'Unsupported file type. Upload JPEG, PNG, WebP, BMP, or DICOM files.' });
        blocking = true;
      }

      // For non-image files (DICOM or blocked types), skip canvas checks
      if (isDicom || !isAllowedType || blocking) {
        resolve({ issues, blocking });
        return;
      }

      // 3 & 4. Load image to check resolution and blur
      const img = new window.Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        // Resolution check
        if (img.width < MIN_RESOLUTION || img.height < MIN_RESOLUTION) {
          issues.push({ type: 'error', message: `Image resolution too low (${img.width}×${img.height}). Minimum ${MIN_RESOLUTION}×${MIN_RESOLUTION} pixels required.` });
          blocking = true;
        }

        // Blur detection via canvas
        if (!blocking) {
          try {
            const canvas = document.createElement('canvas');
            const size = Math.min(img.width, img.height, 256);
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d');
            if (ctx) {
              ctx.drawImage(img, 0, 0, size, size);
              const imgData = ctx.getImageData(0, 0, size, size);
              const lapVar = computeLaplacianVariance(imgData);
              if (lapVar < 80) {
                issues.push({ type: 'warning', message: 'Low image quality detected. Results may be less accurate.' });
              }
            }
          } catch {
            // canvas security errors — ignore
          }
        }

        URL.revokeObjectURL(url);
        resolve({ issues, blocking });
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        issues.push({ type: 'error', message: 'Could not load image for validation.' });
        resolve({ issues, blocking: true });
      };
      img.src = url;
    });
  }, []);

  const handleFileSelect = useCallback(async (file: File) => {
    setValidationIssues([]);
    setPendingFile(null);

    const { issues, blocking } = await validateFile(file);
    setValidationIssues(issues);

    const hasWarning = issues.some((i) => i.type === 'warning');

    if (blocking) {
      // Do NOT proceed — show errors
      return;
    }

    if (hasWarning) {
      // Show warning + "Proceed anyway" button
      setPendingFile(file);
      return;
    }

    // All clear — proceed
    proceedWithFile(file);
  }, [validateFile]);

  const proceedWithFile = useCallback((file: File) => {
    setValidationIssues([]);
    setPendingFile(null);
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
  }, [onImageSelect]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleClear = () => {
    setPreview(null);
    setValidationIssues([]);
    setPendingFile(null);
    onClear();
  };

  return (
    <div className="w-full">
      {!selectedImage ? (
        <>
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

          {/* ── Feature 5 — Validation banners ──────────────────────── */}
          {validationIssues.length > 0 && (
            <div className="mt-3 space-y-2">
              {validationIssues.map((issue, idx) => (
                <div
                  key={idx}
                  className={`flex items-start gap-2 px-3 py-2.5 rounded-xl text-xs md:text-sm ${issue.type === 'error'
                      ? 'bg-red-500/10 border border-red-500/30 text-red-300'
                      : 'bg-amber-500/10 border border-amber-500/30 text-amber-300'
                    }`}
                >
                  {issue.type === 'error' ? (
                    <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  )}
                  <span>{issue.message}</span>
                </div>
              ))}

              {/* "Proceed anyway" for warnings only */}
              {pendingFile && !validationIssues.some((i) => i.type === 'error') && (
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="rounded-lg text-xs border-amber-500/40 text-amber-300 hover:bg-amber-500/10"
                    onClick={() => proceedWithFile(pendingFile)}
                  >
                    <CheckCircle className="w-3.5 h-3.5 mr-1.5" />
                    Proceed anyway
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="rounded-lg text-xs"
                    onClick={() => { setValidationIssues([]); setPendingFile(null); }}
                  >
                    Cancel
                  </Button>
                </div>
              )}
            </div>
          )}
        </>
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
