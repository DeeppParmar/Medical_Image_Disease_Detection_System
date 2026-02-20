import { Activity, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-6 md:py-8 border-t border-border/30 bg-card/30 mt-auto">
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex flex-col items-center gap-4 text-center">
          {/* Logo */}
          <div className="flex items-center gap-2 md:gap-3">
            <div className="p-1.5 md:p-2 rounded-lg bg-secondary border border-border">
              <Activity className="w-4 h-4 md:w-5 md:h-5 text-primary" />
            </div>
            <div>
              <p className="font-display font-bold text-sm md:text-base">
                MediScan <span className="text-primary">AI</span>
              </p>
            </div>
          </div>

          {/* Built with */}
          <p className="text-xs md:text-sm text-muted-foreground flex items-center gap-1.5 justify-center flex-wrap">
            Built with <Heart className="w-3 h-3 text-destructive fill-destructive animate-pulse" />
          </p>

          {/* Bottom row */}
          <div className="flex flex-col sm:flex-row items-center gap-2 sm:gap-4 text-xs text-muted-foreground">
            <p className="uppercase tracking-widest font-semibold">© 2026 MediScan AI</p>
            <div className="hidden sm:block w-px h-3 bg-border/50" />
            <p>For Research & Educational Purposes</p>
            <div className="hidden sm:block w-px h-3 bg-border/50" />
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-green-500">System Online</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
