import { User, Code, Database, Eye, Palette } from "lucide-react";

const team = [
  {
    name: "Nishant Makwana",
    role: "Team Leader | AI/ML Architect",
    icon: User,
  },
  {
    name: "Rajan Parmar",
    role: "Deep Learning Engineer",
    icon: Code,
  },
  {
    name: "Deep Parmar",
    role: "Computer Vision Specialist",
    icon: Eye,
  },
  {
    name: "Hemakshi Rathod",
    role: "Backend & API Developer",
    icon: Database,
  },
  {
    name: "Hetvi Parmar",
    role: "Frontend & UI/UX Designer",
    icon: Palette,
  },
];

const TeamSection = () => {
  return (
    <section className="py-20 relative">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/30 mb-4">
            <span className="font-display font-semibold gradient-text">Team Phoenix</span>
          </div>
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
            Meet Our Team
          </h2>
          <p className="text-muted-foreground">
            Passionate Innovators Driving Change At DA-IICT 2026 Hackathon
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-4 max-w-4xl mx-auto">
          {team.map((member, index) => (
            <div
              key={member.name}
              className="glass-card rounded-xl p-4 flex items-center gap-3 hover:border-primary/50 transition-all duration-300 min-w-[250px]"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="p-2 rounded-lg bg-primary/10">
                <member.icon className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="font-semibold text-sm">{member.name}</p>
                <p className="text-xs text-muted-foreground">{member.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TeamSection;
