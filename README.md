# Fantasy F1 Optimizer

A machine learning-powered Fantasy F1 team optimizer that uses real F1 race data and live Fantasy F1 prices to predict driver cost-effectiveness and help you build the optimal fantasy team.

## �� Features

- **Real F1 Data**: Uses actual race results from 2023-2024 F1 seasons
- **Live Fantasy F1 Prices**: Integration with official Fantasy F1 API
- **Active Drivers Only**: Filters out retired drivers automatically
- **Cost-Effectiveness Predictions**: ML model predicts fantasy points per dollar
- **Team Optimization**: Recommends optimal driver combinations within budget
- **Real-time Ready**: Live price updates and market analysis

## �� **MAJOR UPDATE: Fantasy F1 API Integration**

We now have **official Fantasy F1 API integration**! The system can fetch live driver prices directly from the Fantasy F1 platform.

**API Documentation**: [F1 Fantasy API](https://documenter.getpostman.com/view/11462073/TzY68Dsi)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MySQL database
- Fantasy F1 account (optional, for live prices)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fantasy-F1-Optimizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up database**
   ```bash
   # Create MySQL database
   mysql -u root -p < database_setup.sql
   ```

4. **Configure environment**
   ```bash
   # Create .env file with your database credentials
   cp .env.template .env
   # Edit .env with your database credentials
   ```

5. **Train the model with real F1 data**
   ```bash
   python3 retrain_with_real_data.py
   ```

6. **Run the optimizer**
   ```bash
   # With Fantasy F1 API integration (recommended)
   python3 example_usage_with_fantasy_api.py
   
   # Or basic version
   python3 example_usage.py
   ```

## 📊 Current Status

- ✅ **Real F1 Data**: Model trained on 174 real race records from 9 F1 races
- ✅ **Active Drivers**: 18 current F1 drivers (2024 season)
- ✅ **Fantasy F1 API**: Live price integration ready
- ✅ **Model Performance**: MSE=0.0574, MAE=0.2038
- ✅ **Production Ready**: No sample data, only real race results

## 🏗️ Project Structure

```
Fantasy-F1-Optimizer/
├── app/
│   ├── ComprehensiveF1DataFetcher.py  # Fetches real F1 data from API
│   ├── RealDataDatabase.py            # Database operations for real data
│   ├── DataPipelineAndModel.py        # Core ML model and training
│   └── FantasyF1API.py                # Fantasy F1 API integration
├── data/                              # Preprocessing objects
├── models/                            # Trained ML models
├── example_usage.py                   # Basic application
├── example_usage_with_fantasy_api.py  # With Fantasy F1 API
├── retrain_with_real_data.py          # Model retraining script
├── setup.sh                           # Easy setup script
├── requirements.txt                   # Python dependencies
├── database_setup.sql                 # Database schema
├── .env.template                      # Environment template
├── README.md                          # This file
├── NEXT_STEPS.md                      # Development roadmap
├── FANTASY_F1_API_SETUP.md           # Fantasy F1 API setup guide
└── PROJECT_STATUS.md                  # Current status
```

## 🎯 Usage

### Basic Usage (Mock Prices)
```bash
python3 example_usage.py
```

### With Fantasy F1 API (Live Prices)
```bash
python3 example_usage_with_fantasy_api.py
```

### Retrain Model
```bash
python3 retrain_with_real_data.py
```

## 🔮 Next Steps

### Immediate (High Priority)
1. **✅ Fantasy F1 API Integration** - COMPLETED!
2. **🔄 Docker Containerization** - Package for easy deployment
3. **🔄 Web Frontend** - Build user interface for team selection

### Medium Priority
4. **🔄 Advanced Features** - Qualifying data, practice sessions, weather
5. **🔄 Performance Optimization** - Caching, async API calls
6. **🔄 Testing Suite** - Unit tests, integration tests

### Future Enhancements
7. **🔄 Mobile App** - React Native or Flutter app
8. **🔄 Social Features** - Share teams, leaderboards
9. **🔄 Analytics Dashboard** - Performance tracking, trends

## 📈 Sample Output

```
🏆 Best Value: Carlos SAINZ (Williams)
   Predicted Cost-Effectiveness: 0.5315
   Price: $20.0M

📊 Selected Team Performance:
   Drivers: 5
   Total Cost: $100.0M
   Predicted Total CE: 2.2630
   Average CE per Driver: 0.4526
```

## 🔑 Fantasy F1 API Setup

To use live prices from the Fantasy F1 API:

1. **Get Fantasy F1 Account**: Sign up at [fantasy.formula1.com](https://fantasy.formula1.com)
2. **Configure Credentials**: Edit `app/FantasyF1API.py` with your email/password
3. **Test Integration**: Run `python3 app/FantasyF1API.py`
4. **Use Live Prices**: Run `python3 example_usage_with_fantasy_api.py`

See `FANTASY_F1_API_SETUP.md` for detailed setup instructions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏎️ Data Sources

- **F1 Race Data**: [OpenF1 API](https://api.openf1.org/)
- **Fantasy F1 Prices**: [Official Fantasy F1 API](https://documenter.getpostman.com/view/11462073/TzY68Dsi)
- **Driver Information**: Real F1 race results and positions
- **Fantasy Points**: Calculated using official F1 Fantasy rules

---

*Built with ❤️ for F1 fans and fantasy players*
