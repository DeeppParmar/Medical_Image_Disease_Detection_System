import { Brain, Scan, Shield, Clock } from "lucide-react";

const stats = [
  { icon: Brain, label: "AI Powered", value: "Deep Learning" },
  { icon: Scan, label: "Accuracy", value: "95%+" },
  { icon: Clock, label: "Processing", value: "<3 Sec" },
  { icon: Shield, label: "Diseases", value: "3 Types" },
];

const HeroSection = () => {
  return (
    <section className="relative pt-32 pb-16 overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 hero-gradient" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] opacity-30">
        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-primary/20 to-transparent blur-3xl animate-pulse-slow" />
      </div>
      
      {/* Grid Pattern */}
      <div className="absolute inset-0 opacity-[0.02]" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
      }} />

      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center max-w-4xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/30 mb-6 animate-fade-in">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm font-medium text-primary">AI-Powered Medical Imaging Analysis</span>
          </div>
          
          <h1 className="font-display text-4xl md:text-6xl lg:text-7xl font-bold mb-6 animate-fade-in" style={{ animationDelay: '0.1s' }}>
            Detect Diseases
            <br />
            <span className="gradient-text">Early & Accurately</span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto animate-fade-in" style={{ animationDelay: '0.2s' }}>
            Upload Medical Images For Instant AI Analysis. Detect Tuberculosis, Pneumonia, And Bone Fractures With Explainable Results.
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto animate-fade-in" style={{ animationDelay: '0.3s' }}>
          {stats.map((stat, index) => (
            <div
              key={stat.label}
              className="glass-card rounded-xl p-4 text-center group hover:border-primary/50 transition-all duration-300"
              style={{ animationDelay: `${0.4 + index * 0.1}s` }}
            >
              <div className="inline-flex p-2 rounded-lg bg-primary/10 mb-3 group-hover:bg-primary/20 transition-colors">
                <stat.icon className="w-5 h-5 text-primary" />
              </div>
              <div className="font-display font-bold text-lg">{stat.value}</div>
              <div className="text-xs text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
