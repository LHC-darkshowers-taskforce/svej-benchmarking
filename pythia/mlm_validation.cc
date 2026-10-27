// mlm_validation.cc
// -----------------
// MLM merging-scale validation for the merged t-channel sample
// (p p > qd qd + 0,1,2 j, ickkw = 1).  Showers an LHE file with the point's
// pythia_card.dat and an overridden JetMatching:qCut, and writes, per
// accepted event, the number of ME light partons and the differential jet
// rates d01, d12, d23 (exclusive kT, R = 1, as JetMatching:coneRadius).
//
// MPI and hadronisation are switched off: the MLM veto acts on the parton
// shower before either, so the matched cross section is unchanged, and the
// DJRs are computed from SM-coloured final-state partons only (dark-sector
// partons carry no SM colour and are excluded, as in the matching itself).
//
// Usage:
//   mlm_validation <pythia_card.dat> <events.lhe[.gz]> <qCut> <out.csv> [seed]

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/CombineMatchingInput.h"

using namespace Pythia8;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <pythia_card.dat> <events.lhe[.gz]> <qCut> <out.csv> [seed]\n";
    return 1;
  }
  const std::string card = argv[1];
  const std::string lhe  = argv[2];
  const double qCut      = std::atof(argv[3]);
  const std::string out  = argv[4];
  const int seed = (argc > 5) ? std::atoi(argv[5]) : 12345;

  Pythia pythia;
  pythia.readFile(card);
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe);
  pythia.readString("JetMatching:qCut = " + std::to_string(qCut));
  pythia.readString("PartonLevel:MPI = off");
  pythia.readString("HadronLevel:all = off");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));
  pythia.readString("Next:numberCount = 1000");

  if (!pythia.flag("JetMatching:merge")) {
    std::cerr << "ERROR: card does not enable JetMatching:merge\n";
    return 2;
  }
  CombineMatchingInput matching;
  matching.setHook(pythia);
  if (!pythia.init()) {
    std::cerr << "ERROR: Pythia initialisation failed\n";
    return 2;
  }

  const int nQmatch = pythia.mode("JetMatching:nQmatch");
  // Exclusive kT clustering: power 1, R = 1, no pT cut, |eta| < 10.
  SlowJet slowJet(1, 1.0, 0., 10., 2, 2, nullptr, false);

  std::ofstream csv(out);
  csv << "nme,d01,d12,d23\n";

  std::map<int, int> nPassed;
  int nAccepted = 0, nFailed = 0;
  while (true) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      ++nFailed;
      continue;
    }

    int nLight = 0;
    for (int i = 0; i < pythia.process.size(); ++i) {
      const Particle& p = pythia.process[i];
      if (p.status() == 23 && (p.id() == 21 || p.idAbs() <= nQmatch)) ++nLight;
    }
    ++nPassed[nLight];
    ++nAccepted;

    Event jetInput;
    jetInput.init("jet input", &pythia.particleData);
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (p.isFinal() && p.colType() != 0) jetInput.append(p);
    }
    slowJet.setup(jetInput);
    std::vector<double> steps;
    while (slowJet.sizeAll() - slowJet.sizeJet() > 0) {
      steps.push_back(std::sqrt(slowJet.dNext()));
      slowJet.doStep();
    }
    // Last clustering step = d01, next-to-last = d12, ...
    double d[3] = {-1., -1., -1.};
    for (int k = 0; k < 3; ++k)
      if (int(steps.size()) - 1 - k >= 0) d[k] = steps[steps.size() - 1 - k];
    csv << nLight << "," << d[0] << "," << d[1] << "," << d[2] << "\n";
  }

  pythia.stat();
  std::cout << "\nmlm_validation summary: qCut " << qCut
            << ", accepted " << nAccepted << ", failed " << nFailed
            << ", LHE events read " << pythia.info.nTried() << "\n";
  for (const auto& [n, passed] : nPassed)
    std::cout << "  accepted with " << n << " ME light partons: " << passed << "\n";
  std::cout << "  sigmaGen = " << pythia.info.sigmaGen() * 1e12 << " fb"
            << " +- " << pythia.info.sigmaErr() * 1e12 << " fb\n";
  return 0;
}
