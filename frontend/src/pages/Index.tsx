import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import AnalysisSection from "@/components/AnalysisSection";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="w-full">
        <HeroSection />
        <AnalysisSection />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
