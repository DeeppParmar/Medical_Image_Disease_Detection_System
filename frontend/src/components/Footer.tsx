import { Activity, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 border-t border-border/30">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-primary/10 border border-primary/30">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="font-display font-bold">
                Medi<span className="gradient-text">Scan</span> AI
              </p>
              <p className="text-xs text-muted-foreground">AI That Cares, Technology That Heals</p>
            </div>
          </div>

          <div className="text-center">
            <p className="text-sm text-muted-foreground flex items-center gap-2">
              Built With <Heart className="w-4 h-4 text-destructive fill-destructive" /> By Team Phoenix
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              DA-IICT Hackathon 2026
            </p>
          </div>

          <div className="text-right">
            <p className="text-xs text-muted-foreground">
              © 2026 Team Phoenix. For Educational Purposes Only.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
