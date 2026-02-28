function PhaseFlowTile({ phases, currentPhase, flowPhaseIndex }) {
  return (
    <section className="tile">
      <div className="tile-head">
        <h2>Phase Flow</h2>
        <span className="badge">{currentPhase}</span>
      </div>
      <p className="flow-note">
        Current phase is inferred from live feed. Phases not completed are greyed out.
      </p>
      <div className="tube-flow">
        {phases.map((phase, index) => {
          const done = index <= flowPhaseIndex;
          const active = phase === currentPhase;
          return (
            <div
              key={phase}
              className={["tube-node", done ? "done" : "pending", active ? "active" : ""].join(" ")}
            >
              <div className="dot" />
              <span>{phase}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default PhaseFlowTile;
