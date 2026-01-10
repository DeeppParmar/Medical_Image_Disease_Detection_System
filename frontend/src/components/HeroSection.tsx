import { Brain, Scan, Shield, Clock } from "lucide-react";

const stats = [
  { icon: Brain, label: "AI Powered", value: "Deep Learning" },
  { icon: Scan, label: "Accuracy", value: "95%+" },
  { icon: Clock, label: "Processing", value: "<3 Sec" },
  { icon: Shield, label: "Diseases", value: "3 Types" },
];

const HeroSection = () => {
  return (
    <section className="relative pt-20 md:pt-28 lg:pt-32 pb-12 md:pb-16">
      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="text-center max-w-4xl mx-auto mb-10 md:mb-16">
          <div className="inline-flex items-center gap-2 px-3 md:px-4 py-1.5 md:py-2 rounded-full bg-primary/10 border border-primary/20 mb-4 md:mb-6 animate-pulse-slow">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-xs md:text-sm font-medium text-primary">Powered by Deep Learning</span>
          </div>

          <h1 className="font-display text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mb-4 md:mb-6 tracking-tight leading-tight px-2">
            Advanced Medical Scan
            <span className="block md:inline text-primary italic md:ml-3 mt-2 md:mt-0">Intelligence</span>
          </h1>

          <p className="text-sm md:text-base lg:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed px-4">
            Instant AI-powered detection for Tuberculosis, Pneumonia, and Bone Fractures with clinical-grade accuracy.
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 max-w-3xl mx-auto px-2">
          {stats.map((stat, index) => (
            <div
              key={stat.label}
              className="glass-card rounded-xl p-3 md:p-4 text-center group hover:border-primary/50 transition-all duration-300 hover:scale-105"
            >
              <div className="inline-flex p-1.5 md:p-2 rounded-lg bg-primary/10 mb-2 md:mb-3 group-hover:bg-primary/20 transition-colors">
                <stat.icon className="w-4 h-4 md:w-5 md:h-5 text-primary" />
              </div>
              <div className="font-display font-bold text-base md:text-lg">{stat.value}</div>
              <div className="text-[10px] md:text-xs text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
