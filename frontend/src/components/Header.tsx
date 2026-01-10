import { Activity, Zap } from "lucide-react";

const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-card border-b border-border/30">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full" />
              <div className="relative p-2 rounded-xl bg-primary/10 border border-primary/30">
                <Activity className="w-6 h-6 text-primary" />
              </div>
            </div>
            <div>
              <h1 className="font-display text-xl font-bold tracking-tight">
                Medi<span className="gradient-text">Scan</span> AI
              </h1>
              <p className="text-xs text-muted-foreground">Early Disease Detection</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/30">
              <Zap className="w-3.5 h-3.5 text-primary" />
              <span className="text-xs font-medium text-primary">DA-IICT 2026</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Team</span>
              <span className="font-display font-semibold gradient-text">Phoenix</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
