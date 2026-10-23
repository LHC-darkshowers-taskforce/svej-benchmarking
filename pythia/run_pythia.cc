// run_pythia.cc
// -------------
// Shower and hadronise a MadGraph LHE file with a pythia_card.dat written by
// write_cards.py, and write the events to HepMC3 ASCII.
//
// If the card sets JetMatching:merge = on (the --merged sample), the
// MadGraph-style MLM UserHook is installed here: the setting alone does
// nothing, the hook lives in Pythia8Plugins and must be set by the main
// program.  Vetoed events are skipped inside pythia.next(), so only the
// accepted multiplicities are counted here; the generated ones are in the LHE.
//
// Build (Pythia >= 8.315 with gzip, and HepMC3, installed under $PREFIX):
//   g++ -O2 -std=c++17 run_pythia.cc -o run_pythia \
//       -I$PREFIX/include -L$PREFIX/lib -Wl,-rpath,$PREFIX/lib \
//       -lpythia8 -lHepMC3 -ldl
//
// Usage:
//   run_pythia <pythia_card.dat> <events.lhe[.gz]> <output.hepmc> [seed]

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/CombineMatchingInput.h"
#include "Pythia8Plugins/HepMC3.h"

using namespace Pythia8;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <pythia_card.dat> <events.lhe[.gz]> <output.hepmc> [seed]\n";
    return 1;
  }
  const std::string card = argv[1];
  const std::string lhe  = argv[2];
  const std::string out  = argv[3];
  const int seed = (argc > 4) ? std::atoi(argv[4]) : 12345;

  Pythia pythia;
  pythia.readFile(card);
  // The card's Beams:LHEF is a placeholder; point it at this job's events.
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe);
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));
  pythia.readString("Next:numberCount = 100");

  const bool doMatching = pythia.flag("JetMatching:merge");
  CombineMatchingInput matching;
  if (doMatching) matching.setHook(pythia);

  if (!pythia.init()) {
    std::cerr << "ERROR: Pythia initialisation failed\n";
    return 2;
  }

  Pythia8ToHepMC toHepMC(out);

  // Tally by number of light ME partons (gluons and quarks up to nQmatch).
  const int nQmatch = pythia.mode("JetMatching:nQmatch");
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
    toHepMC.writeNextEvent(pythia);
    ++nAccepted;
  }

  pythia.stat();
  std::cout << "\nrun_pythia summary: accepted " << nAccepted
            << ", failed " << nFailed
            << ", LHE events read " << pythia.info.nTried()
            << (doMatching ? "  (difference = MLM vetoes)" : "") << "\n";
  for (const auto& [n, passed] : nPassed)
    std::cout << "  accepted with " << n << " ME light partons: " << passed << "\n";
  std::cout << "  sigmaGen = " << pythia.info.sigmaGen() * 1e12 << " fb\n";
  return 0;
}
