function DefectCameraSettings({ defectSettings, onFieldChange, onSave, saving }) {
  if (!defectSettings) {
    return <p>Loading defect settings...</p>;
  }

  return (
    <div className="tool-panel">
      <h3>Defect Camera Settings</h3>
      <div className="field-grid">
        <label>
          Source
          <input
            value={defectSettings.source}
            onChange={(e) => onFieldChange("source", e.target.value)}
          />
        </label>
        <label>
          Width
          <input
            type="number"
            value={defectSettings.width}
            onChange={(e) => onFieldChange("width", e.target.value)}
          />
        </label>
        <label>
          Height
          <input
            type="number"
            value={defectSettings.height}
            onChange={(e) => onFieldChange("height", e.target.value)}
          />
        </label>
      </div>
      <label>
        Reference Image Path
        <input
          value={defectSettings.reference_image_path || ""}
          onChange={(e) => onFieldChange("reference_image_path", e.target.value)}
        />
      </label>
      <button onClick={onSave} disabled={saving}>
        {saving ? "Saving..." : "Save Defect Settings"}
      </button>
    </div>
  );
}

export default DefectCameraSettings;
