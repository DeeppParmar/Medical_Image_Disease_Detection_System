import Header from "@/components/Header";
import AnalysisSection from "@/components/AnalysisSection";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      <main className="flex-1 w-full">
        <AnalysisSection />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
