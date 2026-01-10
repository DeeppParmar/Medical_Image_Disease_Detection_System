import { Cpu, Eye, Zap, Shield, Layers, Lightbulb } from "lucide-react";

const features = [
  {
    icon: Cpu,
    title: "Deep Learning CNN",
    description: "ResNet-50 Backbone With Attention Mechanisms For Precise Feature Extraction",
  },
  {
    icon: Eye,
    title: "Contour Mapping",
    description: "Visual Overlays Highlighting Abnormal Regions For Easy Interpretation",
  },
  {
    icon: Lightbulb,
    title: "Explainable AI",
    description: "Grad-CAM Visualization Shows Exactly Why The Model Made Its Decision",
  },
  {
    icon: Zap,
    title: "Real-Time Processing",
    description: "Get Results In Under 3 Seconds With Optimized Inference Pipeline",
  },
  {
    icon: Layers,
    title: "Multi-Disease Detection",
    description: "Simultaneously Detect TB, Pneumonia, And Bone Fractures",
  },
  {
    icon: Shield,
    title: "Clinical-Grade Accuracy",
    description: "95%+ Accuracy Validated Against Expert Radiologist Diagnoses",
  },
];

const FeaturesSection = () => {
  return (
    <section className="py-20 relative">
      <div className="absolute inset-0 opacity-50">
        <div className="absolute top-1/2 left-0 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-primary/5 rounded-full blur-3xl" />
      </div>
      
      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-16">
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
            Powered By <span className="gradient-text">Advanced AI</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            State-Of-The-Art Deep Learning Technology Designed For Resource-Limited Healthcare Settings
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="glass-card rounded-2xl p-6 group hover:border-primary/50 transition-all duration-300 hover:-translate-y-1"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="p-3 rounded-xl bg-primary/10 w-fit mb-4 group-hover:bg-primary/20 transition-colors">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-display font-semibold text-lg mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
