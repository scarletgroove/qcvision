#!/usr/bin/env python3
"""
Utility Scripts Directory
Create this as scripts/utils.py or similar
"""

import pandas as pd
import os
from datetime import datetime


class DefectLogAnalyzer:
    """Analyze defect logs and generate statistics"""
    
    def __init__(self, log_file="defect_log.csv"):
        self.log_file = log_file
        self.df = None
        self.load_log()
    
    def load_log(self):
        """Load defect log CSV"""
        if os.path.exists(self.log_file):
            self.df = pd.read_csv(self.log_file)
            return True
        else:
            print(f"Log file not found: {self.log_file}")
            return False
    
    def get_summary(self):
        """Get summary statistics"""
        if self.df is None or len(self.df) == 0:
            return "No defects logged yet"
        
        summary = {}
        summary['total_defects'] = len(self.df)
        summary['defect_types'] = self.df['Defect_Type'].value_counts().to_dict()
        summary['avg_confidence'] = pd.to_numeric(self.df['Confidence'], errors='coerce').mean()
        summary['min_confidence'] = pd.to_numeric(self.df['Confidence'], errors='coerce').min()
        summary['max_confidence'] = pd.to_numeric(self.df['Confidence'], errors='coerce').max()
        
        return summary
    
    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        
        if isinstance(summary, str):
            print(summary)
            return

        print("\n" + "="*50)
        print("DEFECT INSPECTION SUMMARY")
        print("="*50)
        print(f"\nTotal Defects Found: {summary['total_defects']}")
        print(f"\nDefect Types:")
        for defect_type, count in summary['defect_types'].items():
            percentage = (count / summary['total_defects']) * 100
            print(f"  - {defect_type}: {count} ({percentage:.1f}%)")
        
        print(f"\nConfidence Statistics:")
        print(f"  - Average: {summary['avg_confidence']:.2f}")
        print(f"  - Min: {summary['min_confidence']:.2f}")
        print(f"  - Max: {summary['max_confidence']:.2f}")
        print("\n")
    
    def get_defects_by_type(self, defect_type: str):
        """Get all log rows for a specific defect type."""
        if self.df is None:
            return None
        return self.df[self.df["Defect_Type"] == defect_type]
    
    def export_report(self, output_file=None):
        """Export analysis report"""
        if output_file is None:
            output_file = f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DEFECT INSPECTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            summary = self.get_summary()
            
            if isinstance(summary, str):
                f.write(summary + "\n")
                print(f"Report saved: {output_file}")
                return
            
            f.write(f"Total Defects: {summary['total_defects']}\n\n")
            
            f.write("Defect Types:\n")
            for defect_type, count in summary['defect_types'].items():
                percentage = (count / summary['total_defects']) * 100
                f.write(f"  - {defect_type}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nConfidence Statistics:\n")
            f.write(f"  - Average: {summary['avg_confidence']:.2f}\n")
            f.write(f"  - Min: {summary['min_confidence']:.2f}\n")
            f.write(f"  - Max: {summary['max_confidence']:.2f}\n")
            
            f.write(f"\n\nDetailed Log:\n")
            f.write(self.df.to_string())
        
        print(f"Report saved: {output_file}")


def main():
    """Main entry point"""
    import sys
    
    print("\n📊 Defect Log Analyzer\n")
    
    analyzer = None
    
    # Try to find log file
    log_files = [f for f in os.listdir('.') if f == 'defect_log.csv']
    
    if log_files:
        analyzer = DefectLogAnalyzer(log_files[0])
        analyzer.print_summary()
        
        # Export option
        export = input("Export detailed report? (y/n): ").strip().lower()
        if export == 'y':
            analyzer.export_report()
    else:
        print("❌ No defect log found in current directory")
        print("Run detection first to generate defect_log.csv")


if __name__ == "__main__":
    main()
