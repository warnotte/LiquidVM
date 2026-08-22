/**
 * Point d'entrée du chantier EAU (page eau.html — branche `eau`, non reliée
 * au site public tant qu'aucun jalon n'est mergé, voir PLAN-EAU.md).
 * ÉTAT : J0 — le banc du scatter atomique P2G. Pas encore de simulation :
 * la page mesure et affiche le verdict du critère « < 2 ms/itération ».
 */

import { BenchP2G } from '../../liquid3d/bench_p2g';
import { GRID_EAU, PARTICLES_EAU } from '../../liquid3d/config_eau';
import { acquireDevice } from './gpu';
import { showFatalError } from './overlay';

const J0_BUDGET_MS = 2.0;

async function boot(): Promise<void> {
  const selftest = new URLSearchParams(location.search).has('selftest');
  const hud = document.getElementById('hud-eau') as HTMLDivElement;
  const { device } = await acquireDevice();
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      showFatalError(`Le device GPU a été perdu : ${info.message}`);
    }
  });

  hud.textContent = `banc P2G : ${PARTICLES_EAU.toLocaleString('fr')} particules sur ${GRID_EAU}³, mesure en cours…`;
  const bench = await BenchP2G.create(device);
  const result = await bench.measure();

  // Le verdict J0 porte sur le cas TRIÉ : c'est l'état que le tri périodique
  // (sous-tâche J1, pratique standard FLIP) maintiendra en régime.
  const pass = result.msSorted < J0_BUDGET_MS;
  const fmt = (xs: readonly number[]): string => xs.map((s) => s.toFixed(2)).join(' · ');
  hud.innerHTML =
    `<b>LiquidVM eau — J0 : banc P2G</b><br>` +
    `${PARTICLES_EAU.toLocaleString('fr')} particules groupées → grille MAC ${GRID_EAU}³ ` +
    `(≈ 100 M atomicAdd/itération)<br>` +
    `ordre aléatoire (pire cas) : <b>${result.msRandom.toFixed(2)} ms</b> — ${fmt(result.samplesRandom)}<br>` +
    `ordre trié par cellule : <b>${result.msSorted.toFixed(2)} ms</b> — ${fmt(result.samplesSorted)}<br>` +
    `verdict J0 (cas trié, budget ${J0_BUDGET_MS} ms) : ` +
    `<span class="${pass ? 'pass' : 'fail'}">${pass ? 'PASS' : 'FAIL'}</span>`;

  if (selftest) {
    const report = document.createElement('div');
    report.id = 'selftest';
    report.textContent = JSON.stringify({
      ok: pass,
      msRandom: Number(result.msRandom.toFixed(3)),
      msSorted: Number(result.msSorted.toFixed(3)),
    });
    report.style.display = 'none';
    document.body.appendChild(report);
    document.title = pass ? 'SELFTEST-OK' : 'SELFTEST-FAIL';
  }
}

boot().catch((err: unknown) => {
  showFatalError(err instanceof Error ? err.message : String(err));
  document.title = 'SELFTEST-FAIL';
});
