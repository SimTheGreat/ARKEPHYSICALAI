function pointsToSvg(pointsNorm, frameWidth, frameHeight) {
  return pointsNorm
    .map(([x, y]) => `${Math.round(x * frameWidth)},${Math.round(y * frameHeight)}`)
    .join(" ");
}

function ZoneIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path
        d="M12 2L3 7v10l9 5 9-5V7l-9-5zm0 2.2 6.8 3.8L12 11.8 5.2 8 12 4.2zm-7 5.6 6 3.4v6.3l-6-3.3V9.8zm8 9.7v-6.3l6-3.4v6.4l-6 3.3z"
        fill="currentColor"
      />
    </svg>
  );
}

function TuneIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path
        d="M3 7h8v2H3V7zm10 0h8v2h-8V7zM8 5h2v6H8V5zm6 8h2v6h-2v-6zM3 15h10v2H3v-2zm12 0h6v2h-6v-2z"
        fill="currentColor"
      />
    </svg>
  );
}

function LiveFeedTile({
  currentPhase,
  frameTick,
  liveStatus,
  frameWidth,
  frameHeight,
  visibleZones,
  draftPoints,
  onLiveFrameClick,
  onOpenZoneModal,
  onOpenLiveSettingsModal,
}) {
  return (
    <article className="tile">
      <div className="tile-head">
        <h2>Live Feed: Line Camera</h2>
        <div className="tile-head-actions">
          <button className="icon-btn" onClick={onOpenZoneModal} title="Zone Settings" aria-label="Zone Settings">
            <ZoneIcon />
          </button>
          <button className="icon-btn" onClick={onOpenLiveSettingsModal} title="HSV Settings" aria-label="HSV Settings">
            <TuneIcon />
          </button>
          <span className="badge">{currentPhase}</span>
        </div>
      </div>

      <div className="feed-frame" onClick={onLiveFrameClick} role="button" tabIndex={0}>
        <img
          className="feed-image"
          src={`/api/vision/live/frame?t=${frameTick}`}
          alt="Live line feed"
        />
        <svg
          className="zone-overlay"
          viewBox={`0 0 ${frameWidth} ${frameHeight}`}
          preserveAspectRatio="none"
        >
          {visibleZones.map((zone, idx) => (
            <g key={`${zone.station}-${idx}`}>
              <polygon
                className="zone-polygon"
                points={pointsToSvg(zone.points_norm, frameWidth, frameHeight)}
              />
            </g>
          ))}
          {draftPoints.length > 0 ? (
            <>
              <polyline
                className="zone-draft"
                points={pointsToSvg(draftPoints, frameWidth, frameHeight)}
              />
              {draftPoints.map(([x, y], idx) => (
                <circle
                  key={`draft-${idx}`}
                  cx={Math.round(x * frameWidth)}
                  cy={Math.round(y * frameHeight)}
                  r="4"
                  className="zone-point"
                />
              ))}
            </>
          ) : null}
        </svg>
      </div>
      <p className="click-hint">Click on live feed to add polygon points for selected station.</p>

      {liveStatus ? (
        <div className="status-row">
          <span>Dominant: {liveStatus.dominant_station}</span>
          <span>Score: {Number(liveStatus.dominant_score || 0).toFixed(2)}</span>
          <span>
            Candidates:{" "}
            {liveStatus.instant_candidates?.length
              ? liveStatus.instant_candidates.join(", ")
              : "NOT_PRESENT"}
          </span>
        </div>
      ) : null}
    </article>
  );
}

export default LiveFeedTile;
