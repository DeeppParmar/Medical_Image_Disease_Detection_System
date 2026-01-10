import { Activity, Zap } from "lucide-react";

const Header = () => {
  return (
    <header className="sticky top-0 left-0 right-0 z-50 glass-card border-b border-border/30 backdrop-blur-xl">
      <div className="container mx-auto px-4 md:px-6 py-3 md:py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 md:gap-3">
            <div className="p-1.5 md:p-2 rounded-lg bg-primary/5 border border-primary/20">
              <Activity className="w-4 h-4 md:w-5 md:h-5 text-primary" />
            </div>
            <div>
              <h1 className="font-display text-base md:text-lg font-bold tracking-tight">
                MediScan <span className="text-primary">AI</span>
              </h1>
              <p className="text-[9px] md:text-[10px] text-muted-foreground uppercase tracking-widest font-medium hidden xs:block">
                Early Disease Detection
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 md:gap-6">
            <div className="hidden md:flex items-center gap-4 lg:gap-6 text-sm font-medium text-muted-foreground">
              <a href="#analyze" className="hover:text-primary transition-colors">Analyze</a>
              <a href="#features" className="hover:text-primary transition-colors">Features</a>
            </div>
            <div className="h-4 w-px bg-border/50 hidden md:block" />
            <div className="flex items-center gap-2">
              <span className="px-2 md:px-3 py-1 rounded-full bg-primary/10 text-primary text-[10px] md:text-xs font-bold border border-primary/20">
                BETA
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
