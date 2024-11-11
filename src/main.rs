use std::error::Error;
use chrono::NaiveDateTime;
use plotters::prelude::*;
use csv::Reader;
use std::collections::HashMap;

#[derive(Debug)]
pub struct TradeRecord {
    datetime: NaiveDateTime,
    return_value: f64,
    max_opposite_excursion: Option<f64>,
}

pub fn load_trade_data(file_path: &str, multiplier: Option<f64>) -> Result<Vec<TradeRecord>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(file_path)?;
    let mut trades = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let parsed_return: f64 = record[1].parse()?;
        let parsed_excursion: Option<f64> = record[2].parse().ok();

        let adjusted_return = multiplier.map_or(parsed_return, |m| parsed_return * m);
        let adjusted_excursion = parsed_excursion.map(|exc| multiplier.map_or(exc, |m| exc * m));

        trades.push(TradeRecord {
            datetime: NaiveDateTime::parse_from_str(&record[0], "%Y%m%d %H:%M:%S")?,
            return_value: adjusted_return,
            max_opposite_excursion: adjusted_excursion,
        });
    }
    Ok(trades)
}

// Plotting function
fn plot_data(
    returns: &[f64],
    daily_returns: &[f64],
    trades_per_day: &[usize],
    cumulative_returns: &[f64],
) -> Result<(), Box<dyn Error>> {
    // Plot Cumulative Returns Over Time
    let root = BitMapBackend::new("cumulative_returns.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Cumulative Returns Over Time", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..cumulative_returns.len(), cumulative_returns.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()..cumulative_returns.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())?;

    chart.configure_mesh().draw()?;
    chart
        .draw_series(LineSeries::new(
            cumulative_returns.iter().enumerate().map(|(i, &val)| (i, val)),
            &BLUE,
        ))?
        .label("Cumulative Return")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().background_style(&WHITE).border_style(&BLACK).draw()?;

    // Plot Per Trade Returns Histogram
    let root = BitMapBackend::new("per_trade_returns_histogram.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_return = returns.iter().cloned().fold(0.0 / 0.0, f64::max);
    let min_return = returns.iter().cloned().fold(0.0 / 0.0, f64::min);

    let mut chart = ChartBuilder::on(&root)
        .caption("Per Trade Returns Histogram", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d((min_return as i32 - 1)..(max_return as i32 + 1), 0..returns.len() / 10)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(BLUE.filled())
            .data(returns.iter().map(|x| (*x as i32, 1))),
    )?;

    // Plot Daily Returns Histogram with Bucketing for f64
    let root = BitMapBackend::new("daily_returns_histogram.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let num_buckets = 10;  // Define smaller bucket size by increasing the number of buckets
    let max_daily_return = daily_returns.iter().cloned().fold(0.0 / 0.0, f64::max);
    let min_daily_return = daily_returns.iter().cloned().fold(0.0 / 0.0, f64::min);
    let bucket_width = (max_daily_return - min_daily_return) / num_buckets as f64;

    // Count occurrences per bucket
    let mut bucket_counts = vec![0; num_buckets];
    for &value in daily_returns {
        let bucket_index = ((value - min_daily_return) / bucket_width).floor() as usize;
        if bucket_index < num_buckets {
            bucket_counts[bucket_index] += 1;
        }
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Daily Returns Histogram (f64 Buckets)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (min_daily_return as i32 - 1)..(max_daily_return as i32 + 1),
            0..*bucket_counts.iter().max().unwrap_or(&1)
        )?;

    chart.configure_mesh().draw()?;
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.filled())
            .data(bucket_counts.iter().enumerate().map(|(i, &count)| ((min_daily_return + i as f64 * bucket_width) as i32, count))),
    )?;

    // Plot Trades Per Day Histogram
    let root = BitMapBackend::new("trades_per_day_histogram.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_trades = *trades_per_day.iter().max().unwrap_or(&1);
    let mut chart = ChartBuilder::on(&root)
        .caption("Trades Per Day Histogram", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_trades, 0..trades_per_day.len() / 10)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(GREEN.filled())
            .data(trades_per_day.iter().map(|&x| (x, 1))),
    )?;

    Ok(())
}


// Analyze trade data
fn analyze_trade_data(trades: &[TradeRecord]) {
    // Per trade statistics
    let returns: Vec<f64> = trades.iter().map(|trade| trade.return_value).collect();

    let mut cumulative_returns = Vec::with_capacity(returns.len());
    let mut cum_sum = 0.0;
    for r in &returns {
        cum_sum += *r;
        cumulative_returns.push(cum_sum);
    }

    println!("--- Per Trade Statistics ---");
    let cum_ret: f64= returns.clone().into_iter().sum();
    println!("Cumulative Return: {:.2}", cum_ret);
    let num_trades = returns.len();
    println!("Number of Trades: {}", num_trades);
    calculate_statistics(&returns);
    
    let total_winning_pnl: f64 = returns.iter().filter(|&&x| x > 0.0).sum();
    let total_losing_pnl: f64 = returns.iter().filter(|&&x| x < 0.0).sum();
    let profit_factor = if total_losing_pnl != 0.0 {
        total_winning_pnl / total_losing_pnl.abs()
    } else {
        0.0
    };

    // Maximum win
    let max_win = returns
    .iter()
    .filter(|&&x| x > 0.0)
    .copied()
    .reduce(f64::max)
    .unwrap_or(0.0);

    // Minimum loss (most negative loss)
    let min_loss = returns
    .iter()
    .filter(|&&x| x < 0.0)
    .copied()
    .reduce(f64::min)
    .unwrap_or(0.0);

    // Win Percentage
    let win_count = returns.iter().filter(|&&x| x > 0.0).count();
    let win_percentage = (win_count as f64 / num_trades as f64) * 100.0;

    // Average Win and Average Loss
    let avg_win = if win_count > 0 {
        total_winning_pnl / win_count as f64
    } else {
        0.0
    };
    let avg_loss = if num_trades - win_count > 0 {
        total_losing_pnl / (num_trades - win_count) as f64
    } else {
        0.0
    };
    // Average Max Opposite Excursion for Winning and Losing Trades
    let avg_max_opposite_excursion_win = trades.iter()
        .filter_map(|trade| if trade.return_value > 0.0 { trade.max_opposite_excursion } else { None })
        .sum::<f64>()
        / win_count as f64;

    let avg_max_opposite_excursion_loss = trades.iter()
        .filter_map(|trade| if trade.return_value < 0.0 { trade.max_opposite_excursion } else { None })
        .sum::<f64>()
        / (num_trades - win_count) as f64;

    println!("Win Percentage: {:.2}%", win_percentage);
    println!("Profit Factor: {:.2}", profit_factor);
    println!("Average Win: {:.2}", avg_win);
    println!("Average Loss: {:.2}", avg_loss);
    println!("Max Win: {:.2}", max_win);
    println!("Max Loss: {:.2}", min_loss);
    println!("MAE for Wins: {:.2}", avg_max_opposite_excursion_win);
    println!("MFE for Losses: {:.2}", avg_max_opposite_excursion_loss);

    // Per day statistics
    let mut daily_returns: HashMap<String, f64> = HashMap::new();
    let mut trades_per_day: HashMap<String, usize> = HashMap::new();
    for trade in trades {
        let date_key = trade.datetime.format("%Y%m%d").to_string();
        *daily_returns.entry(date_key.clone()).or_insert(0.0) += trade.return_value;
        *trades_per_day.entry(date_key).or_insert(0) += 1;
    }
    let num_days = daily_returns.len();
    // Trade count statistics
    let trade_counts: Vec<usize> = trades_per_day.values().cloned().collect();
    let mean_trades_per_day = trade_counts.iter().copied().sum::<usize>() as f64 / num_days as f64;
    let median_trades_per_day = median_usize(&trade_counts);
    let stddev_trades_per_day = standard_deviation_usize(&trade_counts, mean_trades_per_day);

    // Maximum trades per day
    let max_trades_per_day = trade_counts.iter().copied().max().unwrap_or(0);

    // Mode and Mode Frequency
    let mode_trades = mode(&trade_counts);
    let mode_frequency = trade_counts.iter().filter(|&&count| count == mode_trades).count();

    let daily_return_values: Vec<f64> = daily_returns.values().cloned().collect();
    println!("\n--- Trades Per Day Statistics ---");
    
    println!("Average Number of Trades Per Day: {:.2}", num_trades as f64/ num_days as f64);
    println!("Median Number of Trades per Day: {}", median_trades_per_day);
    println!("Standard Deviation of Trades per Day: {:.2}", stddev_trades_per_day);
    println!("Most Common (Mode) Trades per Day: {}", mode_trades);
    println!("Mode Frequency: {} days", mode_frequency);
    println!("Maximum Trades in a Single Day: {}", max_trades_per_day);

    println!("\n--- Daily Return Statistics ---");
    println!("Number of Days: {}", num_days);
    calculate_statistics(&daily_return_values);

    // Calculate daily metrics
    println!("\n--- Daily Metrics ---");
    calculate_daily_metrics(&daily_return_values);

    // Drawdown statistics
    println!("\n--- Drawdown Statistics ---");
    calculate_drawdown_statistics(&returns);
    
    let trades_per_day_vec: Vec<usize> = trades_per_day.values().cloned().collect();
    // Plotting
    if let Err(e) = plot_data(&returns, &daily_return_values, &trades_per_day_vec, &cumulative_returns) {
        eprintln!("Error creating plots: {:?}", e);
    }
}

// Calculate daily metrics (Sharpe, Calmar, Sortino ratios)
fn calculate_daily_metrics(daily_returns: &[f64]) {
    let mean_daily_return = daily_returns.iter().copied().sum::<f64>() / daily_returns.len() as f64;
    let stddev_daily_return = standard_deviation(daily_returns, mean_daily_return);
    let annualized_return = mean_daily_return * 252.0;
    let annualized_volatility = stddev_daily_return * (252.0 as f64).sqrt();

    // Sharpe Ratio
    let sharpe_ratio = if annualized_volatility != 0.0 {
        annualized_return / annualized_volatility
    } else {
        0.0
    };

    // Calmar Ratio
    let max_drawdown = calculate_max_drawdown(daily_returns);
    let calmar_ratio = if max_drawdown != 0.0 {
        annualized_return / max_drawdown
    } else {
        0.0
    };

    // Sortino Ratio (using downside deviation)
    let downside_deviation = calculate_downside_deviation(daily_returns, mean_daily_return);
    let sortino_ratio = if downside_deviation != 0.0 {
        annualized_return / (downside_deviation * (252.0 as f64).sqrt())
    } else {
        0.0
    };

    println!("Annualized Sharpe Ratio: {:.2}", sharpe_ratio);
    println!("Calmar Ratio: {:.2}", calmar_ratio);
    println!("Sortino Ratio: {:.2}", sortino_ratio);
}

// Calculate maximum drawdown
fn calculate_max_drawdown(daily_returns: &[f64]) -> f64 {
    let mut max_drawdown = 0.0;
    let mut peak = 0.0;
    let mut cumulative_returns = 0.0;

    for &r in daily_returns {
        cumulative_returns += r;
        if cumulative_returns > peak {
            peak = cumulative_returns;
        }
        let drawdown = peak - cumulative_returns;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    max_drawdown
}

// Calculate downside deviation for Sortino ratio
fn calculate_downside_deviation(daily_returns: &[f64], mean: f64) -> f64 {
    let negative_deviations: Vec<f64> = daily_returns.iter()
        .filter(|&&x| x < mean)
        .map(|&x| (x - mean).powi(2))
        .collect();
    let variance = negative_deviations.iter().sum::<f64>() / daily_returns.len() as f64;
    variance.sqrt()
}

// Calculate basic statistics
fn calculate_statistics(data: &[f64]) {
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    let median = median(data);
    let mean_ad = mean_absolute_deviation(data, mean);
    let stddev = standard_deviation(data, mean);
    let mad = median_absolute_deviation(data, median);
    let skew = skewness(data, mean, stddev);
    let kurtosis = kurtosis(data, mean, stddev);

    println!("Mean: {:.2}", mean);
    println!("Median: {:.2}", median);
    println!("Mean Absolute Deviation: {:.2}", mean_ad);
    println!("Median Absolute Deviation: {:.2}", mad);
    println!("Standard Deviation: {:.2}", stddev);
    println!("Skewness: {:.2}", skew);
    println!("Kurtosis: {:.2}", kurtosis);
}

// Calculate median
fn median(data: &[f64]) -> f64 {
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted_data.len() / 2;
    if sorted_data.len() % 2 == 0 {
        (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    } else {
        sorted_data[mid]
    }
}

// Calculate mean absolute deviation (Mean AD) from a given mean
fn mean_absolute_deviation(data: &[f64], mean: f64) -> f64 {
    let abs_devs: Vec<f64> = data.iter().map(|&x| (x - mean).abs()).collect();
    abs_devs.iter().sum::<f64>() / abs_devs.len() as f64 // Return the mean of the absolute deviations
}

// Calculate median absolute deviation (MAD)
fn median_absolute_deviation(data: &[f64], med: f64) -> f64 {
    let abs_devs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    median(&abs_devs)
}

// Calculate standard deviation
fn standard_deviation(data: &[f64], mean: f64) -> f64 {
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

// Calculate skewness
fn skewness(data: &[f64], mean: f64, stddev: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - mean) / stddev).powi(3)).sum::<f64>() / n
}

// Calculate kurtosis
fn kurtosis(data: &[f64], mean: f64, stddev: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - mean) / stddev).powi(4)).sum::<f64>() / n - 3.0
}

fn median_usize(data: &[usize]) -> usize {
    let mut sorted_data = data.to_vec();
    sorted_data.sort();
    let mid = sorted_data.len() / 2;
    if sorted_data.len() % 2 == 0 {
        (sorted_data[mid - 1] + sorted_data[mid]) / 2
    } else {
        sorted_data[mid]
    }
}

fn standard_deviation_usize(data: &[usize], mean: f64) -> f64 {
    let variance = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

fn mode(data: &[usize]) -> usize {
    let mut occurrences = HashMap::new();
    for &value in data {
        *occurrences.entry(value).or_insert(0) += 1;
    }
    occurrences.into_iter().max_by_key(|&(_, count)| count).map(|(val, _)| val).unwrap_or(0)
}

// Calculate drawdown statistics
fn calculate_drawdown_statistics(returns: &[f64]) {
    let mut max_drawdown = 0.0;
    let mut peak = 0.0;
    let mut cumulative_returns = 0.0;
    let mut drawdowns = Vec::new();

    for &r in returns {
        cumulative_returns += r;
        if cumulative_returns > peak {
            peak = cumulative_returns;
        }
        let drawdown = peak - cumulative_returns;
        drawdowns.push(drawdown);
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    let mean_drawdown = drawdowns.iter().copied().sum::<f64>() / drawdowns.len() as f64;
    println!("Max Drawdown: {:.2}", max_drawdown);
    println!("Expected Drawdown (Mean): {:.2}", mean_drawdown);
}

fn main() {
    let file_path = "./sample_trades.csv";
    match load_trade_data(file_path, Some(20.0)) {
        Ok(trades) => analyze_trade_data(&trades),
        Err(e) => eprintln!("Error loading trade data: {:?}", e),
    }
}
