import { Activity, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-8 md:py-12 border-t border-border/30 bg-card/30">
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 md:gap-6">
          <div className="flex items-center gap-2 md:gap-3">
            <div className="p-1.5 md:p-2 rounded-lg bg-secondary border border-border">
              <Activity className="w-4 h-4 md:w-5 md:h-5 text-primary" />
            </div>
            <div>
              <p className="font-display font-bold text-xs md:text-sm">
                MediScan <span className="text-primary">AI</span>
              </p>
              <p className="text-[9px] md:text-[10px] text-muted-foreground uppercase tracking-wider">Clinical Image Analysis</p>
            </div>
          </div>

          <div className="text-center">
            <p className="text-[10px] md:text-xs text-muted-foreground flex items-center gap-1.5 justify-center">
              Built with <Heart className="w-3 h-3 text-destructive fill-destructive animate-pulse" /> at DICT Hackathon
            </p>
            <p className="text-[9px] text-muted-foreground/60 mt-1">For Research & Educational Purposes</p>
          </div>

          <div className="text-center md:text-right">
            <p className="text-[9px] md:text-[10px] text-muted-foreground uppercase tracking-widest font-semibold">
              © 2026 MediScan AI
            </p>
            <div className="flex items-center gap-1.5 justify-center md:justify-end mt-1">
              <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
              <span className="text-[9px] text-success">System Online</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
