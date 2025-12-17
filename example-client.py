#!/usr/bin/env python3
"""
Example usage and integration tests for forecasting service.
Demonstrates all endpoints and use cases.
"""

import asyncio
import httpx
import json
from typing import Dict, Any
from datetime import datetime, timedelta

# Configuration
BASE_URL = "http://localhost:8001"
STOCKS = ["AAPL", "GOOGL", "MSFT", "TSLA", "META"]


class ForecastingClient:
    """Client for forecasting service"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        response = await self.client.get(f"{self.base_url}/")
        return response.json()
    
    async def forecast(
        self,
        series_id: str,
        method: str = "lstm",
        horizon: int = 30
    ) -> Dict[str, Any]:
        """Generate forecast"""
        payload = {
            "series_id": series_id,
            "method": method,
            "horizon": horizon
        }
        response = await self.client.post(
            f"{self.base_url}/forecast",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def compare_methods(
        self,
        series_id: str,
        horizon: int = 30
    ) -> Dict[str, Any]:
        """Compare LSTM and Prophet"""
        payload = {
            "series_id": series_id,
            "horizon": horizon
        }
        response = await self.client.post(
            f"{self.base_url}/forecast/compare",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def train_model(
        self,
        series_id: str,
        method: str
    ) -> Dict[str, Any]:
        """Train model"""
        payload = {
            "series_id": series_id,
            "method": method
        }
        response = await self.client.post(
            f"{self.base_url}/models/train",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_model_status(
        self,
        series_id: str
    ) -> Dict[str, Any]:
        """Get model status"""
        response = await self.client.get(
            f"{self.base_url}/models/status/{series_id}"
        )
        response.raise_for_status()
        return response.json()


async def example_health_check():
    """Example: Check service health"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Health Check")
    print("="*60)
    
    async with ForecastingClient() as client:
        status = await client.health_check()
        print(json.dumps(status, indent=2))


async def example_single_forecast():
    """Example: Generate single forecast"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Generate LSTM Forecast")
    print("="*60)
    
    async with ForecastingClient() as client:
        print("\nGenerating 30-day LSTM forecast for AAPL...")
        forecast = await client.forecast("AAPL", method="lstm", horizon=30)
        
        print(f"Series: {forecast['series_id']}")
        print(f"Method: {forecast['method']}")
        print(f"Horizon: {forecast['horizon']} days")
        print(f"RMSE: {forecast['rmse']:.4f}")
        print(f"MAPE: {forecast['mape']:.4f}")
        print(f"\nFirst 5 forecast points:")
        
        for i, point in enumerate(forecast['points'][:5]):
            print(f"  Day {i+1}: {point['value']:.2f} "
                  f"(±{(point['upper_bound']-point['value']):.2f})")


async def example_compare_methods():
    """Example: Compare LSTM vs Prophet"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Compare LSTM vs Prophet")
    print("="*60)
    
    async with ForecastingClient() as client:
        print("\nComparing forecasting methods for GOOGL...")
        comparison = await client.compare_methods("GOOGL", horizon=30)
        
        lstm = comparison['lstm_forecast']
        prophet = comparison['prophet_forecast']
        
        print(f"\nLSTM:")
        print(f"  RMSE: {lstm['rmse']:.4f}")
        print(f"  MAPE: {lstm['mape']:.4f}")
        print(f"  30-day forecast: {lstm['points'][-1]['value']:.2f}")
        
        print(f"\nProphet:")
        print(f"  RMSE: {prophet['rmse']:.4f}")
        print(f"  MAPE: {prophet['mape']:.4f}")
        print(f"  30-day forecast: {prophet['points'][-1]['value']:.2f}")
        
        print(f"\n📊 RECOMMENDATION:")
        print(f"  {comparison['recommendation']}")
        
        print(f"\n📈 Analysis:")
        for key, value in comparison['analysis'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


async def example_train_model():
    """Example: Train and cache model"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Train Prophet Model")
    print("="*60)
    
    async with ForecastingClient() as client:
        print("\nTraining Prophet model for MSFT...")
        metrics = await client.train_model("MSFT", "prophet")
        
        print(f"Series: {metrics['series_id']}")
        print(f"Method: {metrics['method']}")
        print(f"Status: {metrics['status']}")
        print(f"\nPerformance:")
        print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
        print(f"  Train MAPE: {metrics['train_mape']:.4f}")
        print(f"  Test MAPE:  {metrics['test_mape']:.4f}")
        print(f"  Training samples: {metrics['training_samples']}")


async def example_model_status():
    """Example: Check model training status"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Check Model Status")
    print("="*60)
    
    async with ForecastingClient() as client:
        print("\nChecking model status for AAPL...")
        status = await client.get_model_status("AAPL")
        
        print(f"Series: {status['series_id']}\n")
        
        if status['lstm_status']:
            lstm = status['lstm_status']
            print("LSTM Model:")
            print(f"  Status: {lstm['status']}")
            print(f"  Test RMSE: {lstm['test_rmse']:.4f}")
            print(f"  Test MAPE: {lstm['test_mape']:.4f}")
            print(f"  Last trained: {lstm['last_trained']}")
        else:
            print("LSTM Model: Not trained")
        
        print()
        
        if status['prophet_status']:
            prophet = status['prophet_status']
            print("Prophet Model:")
            print(f"  Status: {prophet['status']}")
            print(f"  Test RMSE: {prophet['test_rmse']:.4f}")
            print(f"  Test MAPE: {prophet['test_mape']:.4f}")
            print(f"  Last trained: {prophet['last_trained']}")
        else:
            print("Prophet Model: Not trained")


async def example_batch_forecasting():
    """Example: Batch forecast multiple stocks"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Forecasting")
    print("="*60)
    
    async with ForecastingClient() as client:
        print(f"\nForecasting {len(STOCKS)} stocks...")
        
        results = {}
        for stock in STOCKS:
            try:
                forecast = await client.forecast(stock, "lstm", horizon=7)
                results[stock] = {
                    "forecast": forecast['points'][-1]['value'],
                    "rmse": forecast['rmse'],
                    "mape": forecast['mape']
                }
                print(f"  ✓ {stock}: {forecast['points'][-1]['value']:.2f} "
                      f"(RMSE: {forecast['rmse']:.4f})")
            except Exception as e:
                print(f"  ✗ {stock}: {str(e)}")
        
        print("\nBatch Results:")
        print(json.dumps(results, indent=2))


async def example_forecast_comparison_table():
    """Example: Create comparison table"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Forecast Comparison Table")
    print("="*60)
    
    async with ForecastingClient() as client:
        print(f"\nForecasting for 3 stocks with both methods...\n")
        print(f"{'Stock':<8} {'LSTM RMSE':<12} {'Prophet RMSE':<12} {'Better':<10}")
        print("-" * 50)
        
        for stock in ["AAPL", "GOOGL", "MSFT"]:
            try:
                comparison = await client.compare_methods(stock, horizon=7)
                lstm_rmse = comparison['lstm_forecast']['rmse']
                prophet_rmse = comparison['prophet_forecast']['rmse']
                better = comparison['analysis']['method_advantage']
                
                print(f"{stock:<8} {lstm_rmse:<12.4f} {prophet_rmse:<12.4f} {better:<10}")
            except Exception as e:
                print(f"{stock:<8} Error: {str(e)}")


async def example_error_handling():
    """Example: Error handling"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Error Handling")
    print("="*60)
    
    async with ForecastingClient() as client:
        # Invalid method
        print("\nTesting invalid method...")
        try:
            await client.forecast("AAPL", method="invalid_method")
        except Exception as e:
            print(f"  Expected error: {e}")
        
        # Invalid series
        print("\nTesting non-existent series...")
        try:
            await client.forecast("NONEXISTENT", method="lstm")
        except Exception as e:
            print(f"  Expected error: {e}")
        
        # Invalid horizon
        print("\nTesting invalid horizon...")
        try:
            await client.forecast("AAPL", horizon=0)
        except Exception as e:
            print(f"  Expected error: {e}")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("FORECASTING SERVICE - EXAMPLES")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Run examples
        await example_health_check()
        await example_single_forecast()
        await example_compare_methods()
        await example_train_model()
        await example_model_status()
        await example_batch_forecasting()
        await example_forecast_comparison_table()
        await example_error_handling()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())