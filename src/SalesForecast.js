import { useEffect, useState } from 'react';
import Papa from 'papaparse';

const SalesForecast = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        Papa.parse('csv', {
            download: true,
            header: true,
            complete: (result) => {
                setData(result.data);
            },
        });
    }, []);
}

export default SalesForecast;

