import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import AnalysisSection from "@/components/AnalysisSection";
import FeaturesSection from "@/components/FeaturesSection";
import TeamSection from "@/components/TeamSection";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
        <AnalysisSection />
        <FeaturesSection />
        <TeamSection />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
