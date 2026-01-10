import { Cpu, Eye, Zap, Shield, Layers, Lightbulb } from "lucide-react";

const features = [
  {
    icon: Cpu,
    title: "AI Analysis",
    description: "Deep learning models for precise feature extraction across medical imaging.",
  },
  {
    icon: Lightbulb,
    title: "Explainable AI",
    description: "Visualization shows exactly why the model made its decision.",
  },
  {
    icon: Zap,
    title: "Fast Results",
    description: "Get analysis in seconds with our optimized inference pipeline.",
  },
  {
    icon: Shield,
    title: "Multi-Disease",
    description: "Detect Tuberculosis, Pneumonia, and Bone Abnormalities in one platform.",
  },
];

const FeaturesSection = () => {
  return (
    <section className="py-12 md:py-16 lg:py-20 border-t border-border/50" id="features">
      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="text-center mb-10 md:mb-16">
          <h2 className="font-display text-2xl md:text-3xl lg:text-4xl font-bold mb-3 md:mb-4 tracking-tight px-2">
            Integrated <span className="text-primary">Intelligence</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto text-xs md:text-sm px-4">
            Our platform utilizes standardized deep learning architectures for clinical diagnostic support.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6 max-w-6xl mx-auto px-2">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="glass-card rounded-xl md:rounded-2xl p-4 md:p-6 group hover:border-primary/50 transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:shadow-primary/10"
            >
              <div className="p-2 md:p-3 rounded-xl bg-primary/10 w-fit mb-3 md:mb-4 group-hover:bg-primary/20 transition-colors">
                <feature.icon className="w-5 h-5 md:w-6 md:h-6 text-primary" />
              </div>
              <h3 className="font-display font-semibold text-base md:text-lg mb-2">{feature.title}</h3>
              <p className="text-xs md:text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
