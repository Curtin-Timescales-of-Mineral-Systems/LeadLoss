print("Starting test...")
import pdb
import unittest
from unittest.mock import Mock, patch
from application import LeadLossApplication

class TestLeadLossApplication(unittest.TestCase):
    @patch('application.csvUtils.write_output')
    @patch('application.LeadLossView')
    @patch('application.LeadLossModel')

    def test_exportAllAges(self, mock_model_class, mock_view_class, mock_write_output):
            # Arrange
            app = LeadLossApplication()
            mock_view = mock_view_class.return_value
            mock_model = mock_model_class.return_value

            mock_view.getAllAgesOutputFile.return_value = 'output.csv'
            mock_model.samples = [
                Mock(sample_name='Sample1', monteCarloRuns=[
                    Mock(statistics_by_pb_loss_age={1: Mock(score=0.5)}),
                    Mock(statistics_by_pb_loss_age={2: Mock(score=0.6)})
                ]),
                Mock(sample_name='Sample2', monteCarloRuns=[
                    Mock(statistics_by_pb_loss_age={3: Mock(score=0.7)}),
                    Mock(statistics_by_pb_loss_age={4: Mock(score=0.8)})
                ])
            ]

            # Act
            app.exportAllAges()

        # Assert
            mock_write_output.assert_called_once_with(
                ['Sample Name', 'Age', 'Score'],
                [
                    ['Sample1', 1, 0.5],
                    ['Sample1', 2, 0.6],
                    ['Sample2', 3, 0.7],
                    ['Sample2', 4, 0.8]
                ],
                'output.csv'
            )

if __name__ == '__main__':
    unittest.main()