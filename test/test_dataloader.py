
from src.preprocess import get_data

class TestDataLoader:
    def test_dataloader(self):
        # Test the data loader
        project = "ffmpeg"
        devign = get_data(project)
        assert len(devign) > 0, "Data loader returned no data"
        assert "sha_id" in devign.columns, "Data loader did not return sha_id column"
        assert "project" in devign.columns, "Data loader did not return project column"
        assert "vulnerability" in devign.columns, "Data loader did not return vulnerability column"
