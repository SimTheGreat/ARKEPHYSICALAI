function DefectCameraSettings({ defectSettings, onFieldChange, onSave, saving }) {
  if (!defectSettings) {
    return <p>Loading defect settings...</p>;
  }

  return (
    <div className="tool-panel">
      <h3>Defect Camera Settings</h3>
      <p className="muted">Tune image preprocessing before defect diffing.</p>
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
        <label>
          Contrast Gain
          <input
            type="number"
            step="0.05"
            min="0.5"
            max="3.0"
            value={defectSettings.contrast_gain ?? 1.35}
            onChange={(e) => onFieldChange("contrast_gain", e.target.value)}
          />
        </label>
        <label>
          Saturation Gain
          <input
            type="number"
            step="0.05"
            min="0.0"
            max="3.0"
            value={defectSettings.saturation_gain ?? 1.25}
            onChange={(e) => onFieldChange("saturation_gain", e.target.value)}
          />
        </label>
        <label>
          Comparison Method
          <select
            value={defectSettings.comparison_method || "hybrid"}
            onChange={(e) => onFieldChange("comparison_method", e.target.value)}
          >
            <option value="hybrid">Hybrid (Recommended)</option>
            <option value="edge_diff">Edge Diff</option>
            <option value="absdiff">Absolute Diff</option>
          </select>
        </label>
        <label>
          Min Defect Area
          <input
            type="number"
            min="50"
            max="20000"
            value={defectSettings.min_area ?? 800}
            onChange={(e) => onFieldChange("min_area", e.target.value)}
          />
        </label>
        <label>
          Fail Ratio
          <input
            type="number"
            step="0.0005"
            min="0.0001"
            max="0.2"
            value={defectSettings.fail_ratio ?? 0.003}
            onChange={(e) => onFieldChange("fail_ratio", e.target.value)}
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
