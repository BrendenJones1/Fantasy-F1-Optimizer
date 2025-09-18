#!/bin/bash

# Fantasy F1 Optimizer Setup Script

echo "🏁 Fantasy F1 Optimizer Setup"
echo "=============================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if MySQL is installed
if ! command -v mysql &> /dev/null; then
    echo "❌ MySQL is not installed. Please install MySQL first."
    exit 1
fi

echo "✅ MySQL found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your database credentials"
    echo "   nano .env"
else
    echo "✅ .env file already exists"
fi

# Check if database exists
echo "🗄️  Setting up database..."
mysql -u root -p < database_setup.sql

if [ $? -eq 0 ]; then
    echo "✅ Database setup completed"
else
    echo "❌ Database setup failed. Please check your MySQL credentials."
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your database credentials"
echo "2. Run: python3 retrain_with_real_data.py"
echo "3. Run: python3 example_usage.py"
echo ""
echo "For more information, see README.md and NEXT_STEPS.md"
