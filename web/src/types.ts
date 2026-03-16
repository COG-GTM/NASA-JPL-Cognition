export interface UseCase {
  id: string;
  title: string;
  situation: string;
  nasaRepo: string;
  nasaRepoUrl: string;
  before: {
    description: string;
    code: string;
    painPoints: string[];
  };
  devinAction: string;
  after: {
    description: string;
    code: string;
    improvements: string[];
  };
  value: string;
  metrics: { label: string; value: string }[];
}

export interface Module {
  id: string;
  title: string;
  subtitle: string;
  icon: string;
  color: string;
  colorLight: string;
  description: string;
  useCases: UseCase[];
}
